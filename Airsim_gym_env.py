# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:05:35 2022

@author: jamyl
"""


import gym
import airsim
import numpy as np
import time
from jamys_toolkit import (
    Circuit_wrapper,
    convert_lidar_data_to_polar,
    lidar_formater,
    normalize_action,
    denormalize_action,
    convert_global_to_relative_position,
)
import matplotlib.pyplot as plt
import cv2
import random
from stable_baselines3 import SAC

ENNEMI_MODEL = SAC.load("Training/Lidar_only")


def proximity_jammer(lidar):
    """ Crop the closest lidar points


    Parameters
    ----------
    lidar : numpy array
        lidar data

    Returns
    -------
    numpy array
        croped lidar

    """
    proximity_index = np.where(lidar[:, 1] <= 150 * 6.25 / 1000)
    return np.delete(lidar, proximity_index, 0)


class BaseAirSimEnv(gym.Env):
    """ Base AirSim environment containing all general methods"""

    def __init__(
        self,
        client,
        lidar_size,
        dt,
        ClockSpeed,
        UE_spawn_point,
        liste_checkpoints_coordinates,
        liste_spawn_point,
        is_rendered=False,
        random_reverse=False,
    ):
        super().__init__()
        self.client = client
        self.car_controls = airsim.CarControls()
        self.car_state = client.getCarState()
        # car_state is an AirSim object that contains informations that are not
        # obtainable in real experiments. Therefore, it cannot be used as a
        # MDP object. However, it is useful for computing the reward

        self.is_rendered = is_rendered
        self.dt = dt
        self.ClockSpeed = ClockSpeed
        self.lidar_size = lidar_size
        self.random_reverse = random_reverse

        self.total_reward = 0
        self.done = False

        self.UE_spawn_point = UE_spawn_point
        self.liste_checkpoints_coordonnes = liste_checkpoints_coordinates
        self.liste_spawn_point = liste_spawn_point

        # ________ init of reset() attributes ___________

        self.reversed_world = None
        self.reversed_world = None
        self.observation = None
        self.Circuit1 = None
        self.ax = None
        self.reward = None
        self.gate_passed = None

    def init_values(self):
        """ Init to be done during reset()
        

        Returns
        -------
        None.

        """

        self.done = False

        self.total_reward = 0
        # let's skip the first frames to inialise lidar and make sure everything is right
        self.client.simContinueForFrames(100)
        # the lidar data can take a bit of time before initialisation.
        time.sleep(1)

        if self.random_reverse:
            self.reversed_world = random.choice([False, True])

        print("reset")

    def airsim_step(self):
        """ Make a simulation step
        

        Returns
        -------
        None.

        """
        self.client.simPause(False)
        # TODO a dedicated thread may be more efficient
        time.sleep(self.dt / self.ClockSpeed)
        self.client.simPause(True)

    def checkpoint_update(self):
        """ Update the checkpoint status.

        Returns
        -------
        None.

        """
        self.car_state = self.client.getCarState()
        position = self.car_state.kinematics_estimated.position
        self.gate_passed = self.Circuit1.cycle_tick(position.x_val, position.y_val)[0]

    def random_respawn(self, vehicle_name=None):
        """ Sample a random respwan point

        Parameters
        ----------
        vehicle_name : Str, optional
            Name of the main agent when there are multiple agents.
            The default is None.

        Returns
        -------
        None.

        """
        # be careful, the call arguments for quaterninons are x,y,z,w
        Circuit_wrapper1 = Circuit_wrapper(
            self.liste_spawn_point,
            self.liste_checkpoints_coordonnes,
            UE_spawn_point=self.UE_spawn_point,
        )
        spawn_point, theta, self.Circuit1 = Circuit_wrapper1.sample_random_spawn_point()

        x_val, y_val, z_val = spawn_point.x, spawn_point.y, spawn_point.z

        pose = airsim.Pose()

        orientation = airsim.Quaternionr(0, 0, np.sin(theta / 2) * 1, np.cos(theta / 2))
        position = airsim.Vector3r(x_val, y_val, z_val)
        pose.position = position
        pose.orientation = orientation
        if vehicle_name is not None:
            self.client.simSetVehiclePose(
                pose, vehicle_name=vehicle_name, ignore_collision=True
            )
        else:
            self.client.simSetVehiclePose(pose, ignore_collision=True)

    def reward_calculation(self):
        """ Computes the reward.

        Returns
        -------
        None.

        """
        collision_info = self.client.simGetCollisionInfo()
        crash = collision_info.has_collided

        reward = 0
        if crash:
            reward = -100

        elif self.car_state.speed <= 1:  # Lets force the car to move
            reward = -0.1

        if self.gate_passed:
            reward += 50
            print("gate_passed")

        self.total_reward = self.total_reward + reward

        if crash:
            self.done = True
            print("Crash occured")
            print("Episode reward : " + str(self.total_reward) + 2 * "\n")

        self.reward = reward

    def close(self):
        self.client.simPause(False)
        self.client.enableApiControl(False)

    def render(self, mode="human"):
        if not self.is_rendered:
            fig = plt.figure()
            self.ax = fig.add_subplot(projection="polar")
            self.is_rendered = True

        self.ax.clear()
        T = self.observation["current_lidar"][:, 0]
        R = self.observation["current_lidar"][:, 1]
        self.ax.scatter(T, R)
        plt.pause(1e-6)
        plt.draw()

    # =============================================================================
    #         # ________________ Image ___________________________________________________
    #         responses = self.client.simGetImages(
    #             [airsim.ImageRequest("Camera1", airsim.ImageType.Scene, False, False)],
    #             "MyVehicle",
    #         )  # scene vision image in uncompressed RGB array
    #         response = responses[0]
    #
    #         # get numpy array
    #         img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    #
    #         # reshape array to 4 channel image array H X W X 4
    #         img_rgb = img1d.reshape(response.height, response.width, 3)
    #
    #         cv2.imshow("image", img_rgb)
    # =============================================================================

    def spawn_points_checker(self, wait_time=1):
        self.close()
        i = 0
        for spawn_point in self.liste_spawn_point:
            print("Spawn : " + str(i) + "\n")
            theta_m = spawn_point.theta_min
            theta_M = spawn_point.theta_max
            x_val, y_val, z_val = spawn_point.x, spawn_point.y, spawn_point.z

            pose = airsim.Pose()
            print("\tTheta min = " + str(theta_m))
            orientation = airsim.Quaternionr(
                0, 0, np.sin(theta_m / 2) * 1, np.cos(theta_m / 2)
            )
            position = airsim.Vector3r(x_val, y_val, z_val)
            pose.position = position
            pose.orientation = orientation
            self.client.simSetVehiclePose(pose, ignore_collision=True)
            time.sleep(wait_time)

            print("\tTheta max = " + str(theta_M))
            orientation = airsim.Quaternionr(
                0, 0, np.sin(theta_M / 2) * 1, np.cos(theta_M / 2)
            )
            position = airsim.Vector3r(x_val, y_val, z_val)
            pose.position = position
            pose.orientation = orientation
            self.client.simSetVehiclePose(pose, ignore_collision=True)
            time.sleep(wait_time)
            i += 1


class BoxAirSimEnv(BaseAirSimEnv):
    """Custom AirSim Environment with Box action space following gym interface"""

    def __init__(
        self,
        client,
        lidar_size,
        dt,
        ClockSpeed,
        UE_spawn_point,
        liste_checkpoints_coordinates,
        liste_spawn_point,
        is_rendered=False,
        random_reverse=False,
    ):
        """


        Parameters
        ----------
        client : TYPE
            AirSim client.
        lidar_size : int
            Number of point observed by the lidar in each observation.
        dt : float
            Simulation time step separating two observations/actions.
        ClockSpeed : int
            ClockSpeed selected in SETTINGS.JSON. Make sure that FPS/ClockSpeed > 30
        UE_spawn_point : numpy array
            coordinates of the "player spawn" object in unreal engine
        liste_checkpoints_coordinates : list [x,y,r]
            list of the coordinates of the checkpoints. In UE metric system.
        is_rendered : Boolean, optional
            Wether or not the lidar map is rendering. The default is False.
        random_reverse : boolean
            whether or not the the observations and action should randomly be reversed.
            Strongly advised for training


        Returns
        -------
        None.

        """
        super().__init__(
            client,
            lidar_size,
            dt,
            ClockSpeed,
            UE_spawn_point,
            liste_checkpoints_coordinates,
            liste_spawn_point,
            is_rendered=False,
            random_reverse=False,
        )

        # __________ Below are MDP related objects ___________

        self.action_space = gym.spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # In this order
        # "throttle"
        # "steering"
        # both actions are normalized in [0,1]
        # Example for using image as input (channel-first; channel-last also works):

        low = np.zeros((lidar_size, 2), dtype=np.float32)
        high = np.zeros((lidar_size, 2), dtype=np.float32)

        low[:, 0] = -np.pi
        low[:, 1] = 0

        high[:, 0] = np.pi
        high[:, 1] = np.inf

        self.observation_space = gym.spaces.Dict(
            spaces={
                # the format is [angle , radius]
                "current_lidar": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_lidar": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_throttle": gym.spaces.Box(low=0, high=1, shape=(1,)),
                "prev_steering": gym.spaces.Box(low=0, high=1, shape=(1,)),
            }
        )

    def step(self, action):
        info = {}  # Just a debugging feature here

        # The actions are extracted from the "action" argument

        denormalized_action = denormalize_action(action)
        self.car_controls.throttle = float(denormalized_action[0])
        self.car_controls.steering = float(denormalized_action[1])

        if self.reversed_world:
            self.car_controls.steering *= -1

        self.client.setCarControls(self.car_controls)

        # Now that everything is good and proper, let's run  AirSim a bit
        self.airsim_step()

        # _______ updating the checkpoint situation ___________________

        self.checkpoint_update()

        # __________ extracts the observation ______________________

        self.observation_maker(init=False)

        # ___________ Updates the reward ____________________

        self.reward_calculation()

        # displays ( or not ) the lidar observation
        if self.is_rendered:
            self.render()

        return self.observation, self.reward, self.done, info

    def reset(self):

        self.client.reset()

        # _______ random respawn ________________________

        self.random_respawn()

        # ______ Init agent ______________________________

        self.car_controls.throttle = 0
        self.car_controls.steering = 0

        self.init_values()

        self.observation_maker(init=True)

        return self.observation  # reward, done, info can't be included

    def observation_maker(self, init=False):
        current_raw_lidar = convert_lidar_data_to_polar(self.client.getLidarData())
        current_raw_lidar = proximity_jammer(current_raw_lidar)
        current_lidar, lidar_error = lidar_formater(current_raw_lidar, self.lidar_size)
        if lidar_error:
            print(
                """"Caution, no point was observed by the lidar, the vehicule may be escaping:
                    reseting sim"""
            )
            # Alas, Done cannot be returned by init, but step() will take care of ending the sim
            self.done = True

        if init:
            prev_lidar = np.copy(current_lidar)
        else:
            prev_lidar = self.observation["current_lidar"]

        prev_throttle = self.car_controls.throttle
        prev_steering = self.car_controls.steering

        if self.reversed_world:
            current_lidar[:, 0] *= -1
            current_lidar = current_lidar[::-1]
            prev_lidar[:, 0] *= -1
            prev_lidar = current_lidar[::-1]

        observation = {
            "current_lidar": current_lidar,
            "prev_lidar": prev_lidar,
            "prev_throttle": prev_throttle,
            "prev_steering": prev_steering,
        }

        self.observation = observation


class BoxAirSimEnv_5_memory(BaseAirSimEnv):
    """Custom AirSim Environment with Box action space that follows gym interface"""

    def __init__(
        self,
        client,
        lidar_size,
        dt,
        ClockSpeed,
        UE_spawn_point,
        liste_checkpoints_coordinates,
        liste_spawn_point,
        is_rendered=False,
        random_reverse=False,
    ):
        """


        Parameters
        ----------
        client : TYPE
            AirSim client.
        lidar_size : int
            Number of point observed by the lidar in each observation.
        dt : float
            Simulation time step separating two observations/actions.
        ClockSpeed : int
            ClockSpeed selected in SETTINGS.JSON. Make sure that FPS/ClockSpeed > 30
        UE_spawn_point : numpy array
            coordinates of the "player spawn" object in unreal engine
        liste_checkpoints_coordinates : list [x,y,r]
            list of the coordinates of the checkpoints. In UE metric system.
        is_rendered : Boolean, optional
            Wether or not the lidar map is rendering. The default is False.
        random_reverse : boolean
            whether or not the the observations and action should randomly be reversed.
            Strongly advised for training


        Returns
        -------
        None.

        """
        super().__init__(
            client,
            lidar_size,
            dt,
            ClockSpeed,
            UE_spawn_point,
            liste_checkpoints_coordinates,
            liste_spawn_point,
            is_rendered=False,
            random_reverse=False,
        )

        # ____________ Below are MDP related objects ______________

        self.action_space = gym.spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # =============================================================================
        #         In this order
        #         "throttle"
        #         "steering"
        #         both actions are normalized in [0,1]
        #         Example for using image as input (channel-first; channel-last also works):
        # =============================================================================

        low = np.zeros((lidar_size, 2), dtype=np.float32)
        high = np.zeros((lidar_size, 2), dtype=np.float32)

        low[:, 0] = -np.pi
        low[:, 1] = 0

        high[:, 0] = np.pi
        high[:, 1] = np.inf

        self.observation_space = gym.spaces.Dict(
            spaces={
                # the format is [angle , radius]
                "current_lidar": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_lidar1": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_lidar2": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_lidar3": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_lidar4": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_lidar5": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_throttle": gym.spaces.Box(low=0, high=1, shape=(1,)),
                "prev_steering": gym.spaces.Box(low=0, high=1, shape=(1,)),
            }
        )

    def step(self, action):
        info = {}  # Just a debugging feature here

        # The actions are extracted from the "action" argument

        denormalized_action = denormalize_action(action)
        self.car_controls.throttle = float(denormalized_action[0])
        self.car_controls.steering = float(denormalized_action[1])

        if self.reversed_world:
            self.car_controls.steering *= -1

        self.client.setCarControls(self.car_controls)

        # Now that everything is good and proper, let's run  AirSim a bit
        self.airsim_step()

        # _______ updating the checkpoint situation ___________________

        self.checkpoint_update()

        # __________ extracts the observation ______________________

        self.observation_maker(init=False)

        # ___________ Updates the reward ____________________

        self.reward_calculation()

        # displays ( or not ) the lidar observation
        if self.is_rendered:
            self.render()

        return self.observation, self.reward, self.done, info

    def reset(self):

        self.client.reset()

        # _______ random respawn ________________________

        self.random_respawn()

        # ______ Init agent ______________________________

        self.car_controls.throttle = 0
        self.car_controls.steering = 0

        self.init_values()

        self.observation_maker(init=True)

        return self.observation  # reward, done, info can't be included

    def observation_maker(self, init=False):
        """ Fetch observation

        Contains all sort of operation on lidar data. Proximity jammer, uniform
        sampling, and a killswitch if no lidar point is detected.
        Parameters
        ----------
        init : Boolean, optional
            Whether or not this observation is called by reset() or step().
            Basically, whether or not the lidar memory is empty.
            The default is False.
        Returns
        -------
        None.
        """

        current_raw_lidar = convert_lidar_data_to_polar(self.client.getLidarData())
        current_raw_lidar = proximity_jammer(
            current_raw_lidar
        )  # removing closest points
        current_lidar, lidar_error = lidar_formater(current_raw_lidar, self.lidar_size)

        if init:  # Initialising memory
            prev_lidar1 = np.copy(current_lidar)
            prev_lidar2 = np.copy(current_lidar)
            prev_lidar3 = np.copy(current_lidar)
            prev_lidar4 = np.copy(current_lidar)
            prev_lidar5 = np.copy(current_lidar)
        else:
            prev_lidar5 = self.observation["prev_lidar4"]
            prev_lidar4 = self.observation["prev_lidar3"]
            prev_lidar3 = self.observation["prev_lidar2"]
            prev_lidar2 = self.observation["prev_lidar1"]
            prev_lidar1 = self.observation["current_lidar"]

        prev_throttle = np.array([self.car_controls.throttle])
        prev_steering = np.array([self.car_controls.steering])

        if lidar_error:
            print(
                """"Caution, no point was observed by the lidar, the vehicule may be escaping:
                    reseting sim"""
            )
            # Alas, Done cannot be returned by init, but step() will take care of ending the sim
            self.done = True

        if self.reversed_world:
            current_lidar[:, 0] *= -1
            current_lidar = current_lidar[::-1]
            prev_lidar1[:, 0] *= -1
            prev_lidar1 = prev_lidar1[::-1]
            prev_lidar2[:, 0] *= -1
            prev_lidar2 = prev_lidar2[::-1]
            prev_lidar3[:, 0] *= -1
            prev_lidar3 = prev_lidar3[::-1]
            prev_lidar4[:, 0] *= -1
            prev_lidar4 = prev_lidar4[::-1]
            prev_lidar5[:, 0] *= -1
            prev_lidar5 = prev_lidar5[::-1]

        observation = {
            "current_lidar": current_lidar,
            "prev_lidar1": prev_lidar1,
            "prev_lidar2": prev_lidar2,
            "prev_lidar3": prev_lidar3,
            "prev_lidar4": prev_lidar4,
            "prev_lidar5": prev_lidar5,
            "prev_throttle": prev_throttle,
            "prev_steering": prev_steering,
        }
        self.observation = observation


class BoxAirSimEnv_MultiAgent(BaseAirSimEnv):
    """Custom AirSim Environment with Box action space that follows gym interface"""

    def __init__(
        self,
        client,
        lidar_size,
        dt,
        ClockSpeed,
        UE_spawn_point,
        liste_checkpoints_coordinates,
        liste_spawn_point,
        is_rendered=False,
        random_reverse=False,
    ):
        """


        Parameters
        ----------
        client : TYPE
            AirSim client.
        lidar_size : int
            Number of point observed by the lidar in each observation.
        dt : float
            Simulation time step separating two observations/actions.
        ClockSpeed : int
            ClockSpeed selected in SETTINGS.JSON. Make sure that FPS/ClockSpeed > 30
        UE_spawn_point : numpy array
            coordinates of the "player spawn" object in unreal engine
        liste_checkpoints_coordinates : list [x,y,r]
            list of the coordinates of the checkpoints. In UE metric system.
        is_rendered : Boolean, optional
            Wether or not the lidar map is rendering. The default is False.
        random_reverse : boolean
            whether or not the the observations and action should randomly be reversed.
            Strongly advised for training


        Returns
        -------
        None.

        """
        super().__init__(
            client,
            lidar_size,
            dt,
            ClockSpeed,
            UE_spawn_point,
            liste_checkpoints_coordinates,
            liste_spawn_point,
            is_rendered=False,
            random_reverse=False,
        )

        self.multi_agent_obs = {}

        self.multi_agent_control = {}
        self.multi_agent_control["A_MyVehicle"] = airsim.CarControls()
        self.ennemi_number = 4
        for i in range(1, self.ennemi_number + 1):
            self.multi_agent_control["Car{}".format(i)] = airsim.CarControls()
            self.multi_agent_control["Car{}".format(i)].throttle = 0
            self.multi_agent_control["Car{}".format(i)].steering = 0

        self.car_state = client.getCarState("A_MyVehicle")

        # ______ Fixed ennemi respawn _____________________________
        spawn1 = convert_global_to_relative_position(
            self.UE_spawn_point, [-12990, 4620, 350]
        )
        angle1 = 0
        respawn1 = {"coordinates": spawn1, "angle": angle1}

        spawn2 = convert_global_to_relative_position(
            self.UE_spawn_point, [-9240, -400, 350]
        )
        angle2 = 0
        respawn2 = {"coordinates": spawn2, "angle": angle2}

        spawn3 = convert_global_to_relative_position(
            self.UE_spawn_point, [-12030, -7000, 350]
        )
        angle3 = 180
        respawn3 = {"coordinates": spawn3, "angle": angle3}

        spawn4 = convert_global_to_relative_position(
            self.UE_spawn_point, [-19360, -4030, 350]
        )
        angle4 = 90
        respawn4 = {"coordinates": spawn4, "angle": angle4}

        self.ennemi_respawn = {1: respawn1, 2: respawn2, 3: respawn3, 4: respawn4}

        # ____________ Below are MDP related objects ______________

        self.action_space = gym.spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # =============================================================================
        #         In this order
        #         "throttle"
        #         "steering"
        #         both actions are normalized in [0,1]
        #         Example for using image as input (channel-first; channel-last also works):
        # =============================================================================

        low = np.zeros((lidar_size, 2), dtype=np.float32)
        high = np.zeros((lidar_size, 2), dtype=np.float32)

        low[:, 0] = -np.pi
        low[:, 1] = 0

        high[:, 0] = np.pi
        high[:, 1] = np.inf

        self.observation_space = gym.spaces.Dict(
            spaces={
                # the format is [angle , radius]
                "current_lidar": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_lidar1": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_lidar2": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_lidar3": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_lidar4": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_lidar5": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_throttle": gym.spaces.Box(low=0, high=1, shape=(1,)),
                "prev_steering": gym.spaces.Box(low=0, high=1, shape=(1,)),
            }
        )

    def step(self, action):
        info = {}  # Just a debugging feature here

        # The actions are extracted from the "action" argument

        denormalized_action = denormalize_action(action)
        self.multi_agent_control["A_MyVehicle"].throttle = float(denormalized_action[0])
        self.multi_agent_control["A_MyVehicle"].steering = float(denormalized_action[1])

        if self.reversed_world:
            self.multi_agent_control["A_MyVehicle"].steering *= -1

        for i in range(1, 5):
            self.decision_maker(i)
        self.action_pusher()

        # Now that everything is good and proper, let's run  AirSim a bit
        self.airsim_step()

        # _______ updating the checkpoint situation ___________________

        self.checkpoint_update()

        # __________ extracts the observation ______________________

        self.car_state = self.client.getCarState("A_MyVehicle")
        for i in range(5):
            self.observation_maker(i)

        # ___________ Updates the reward ____________________

        self.reward_calculation()

        # displays ( or not ) the lidar observation
        if self.is_rendered:
            self.render()

        return self.multi_agent_obs["A_MyVehicle"], self.reward, self.done, info

    def reset(self):

        print("reset")
        self.client.reset()

        # be careful, the call arguments for quaterninons are x,y,z,w

        # _______ main car respawn ______________________________

        self.random_respawn(vehicle_name="A_MyVehicle")

        # _____________Ennemi car respawn _________________

        for i in range(1, 5):
            x_val, y_val, z_val = self.ennemi_respawn[i]["coordinates"]
            theta = self.ennemi_respawn[i]["angle"] * np.pi / 180

            pose = airsim.Pose()

            orientation = airsim.Quaternionr(
                0, 0, np.sin(theta / 2) * 1, np.cos(theta / 2)
            )
            position = airsim.Vector3r(x_val, y_val, z_val)
            pose.position = position
            pose.orientation = orientation
            self.client.simSetVehiclePose(
                pose, ignore_collision=True, vehicle_name="Car{}".format(i)
            )

        # ______ Init agent ______________________________

        self.init_values()

        for i in range(5):
            self.observation_maker(i, init=True)

        return self.multi_agent_obs[
            "A_MyVehicle"
        ]  # reward, done, info can't be included

    def decision_maker(self, i, throttle_penalty=0.6):
        """ Chose an action for actor i

        Compute the self.multi_agent_control therm related to the i agent, with
        a penalty applied on throttle ( we want the learning agent to faster
        than the ennemis to encounter entersetign beahviour)

        Parameters
        ----------
        i : int
            agent ID
        throttle_penalty : float, optional
            Coeffficient to be applied to ennemi throttle. The default is 0.6.

        Returns
        -------
        None.

        """
        if not 1 <= i <= self.ennemi_number:
            raise ValueError(
                """The ennemi ID is not matching with the
                             number of ennemis ( received i={},
                            expecting {} ennemis""".format(
                    i, self.ennemi_number
                )
            )

        denormalized_action = denormalize_action(
            ENNEMI_MODEL.predict(observation=self.multi_agent_obs["Car{}".format(i)])[0]
        )
        self.multi_agent_control["Car{}".format(i)].throttle = (
            float(denormalized_action[0]) * throttle_penalty
        )
        self.multi_agent_control["Car{}".format(i)].steering = float(
            denormalized_action[1]
        )

        if self.reversed_world:
            self.multi_agent_control["Car{}".format(i)].steering *= -1

    def action_pusher(self):
        """ Push the CarControl to AirSim for every agent

        Returns
        -------
        None.

        """
        for agent_name in self.multi_agent_obs.keys():
            self.client.setCarControls(self.multi_agent_control[agent_name], agent_name)

    def observation_maker(self, i, init=False):
        """ Fetch observation for the agent i

        Contains all sort of operation on lidar data. Proximity jammer, uniform
        sampling, and a killswitch if no lidar point is detected.
        Parameters
        ----------
        i : int
            Agent ID where the observation has to be fetched
        init : Boolean, optional
            Whether or not this observation is called by reset() or step().
            Basically, whether or not the lidar memory is empty.
            The default is False.

        Returns
        -------
        None.

        """
        if i == 0:
            name = "A_MyVehicle"
        else:
            name = "Car{}".format(i)

        current_raw_lidar = convert_lidar_data_to_polar(
            self.client.getLidarData(vehicle_name=name)
        )
        current_raw_lidar = proximity_jammer(
            current_raw_lidar
        )  # removing closest points
        current_lidar, lidar_error = lidar_formater(current_raw_lidar, self.lidar_size)

        if init:  # Initialising memory
            prev_lidar1 = np.copy(current_lidar)
            prev_lidar2 = np.copy(current_lidar)
            prev_lidar3 = np.copy(current_lidar)
            prev_lidar4 = np.copy(current_lidar)
            prev_lidar5 = np.copy(current_lidar)
        else:
            prev_lidar5 = self.multi_agent_obs[name]["prev_lidar4"]
            prev_lidar4 = self.multi_agent_obs[name]["prev_lidar3"]
            prev_lidar3 = self.multi_agent_obs[name]["prev_lidar2"]
            prev_lidar2 = self.multi_agent_obs[name]["prev_lidar1"]
            prev_lidar1 = self.multi_agent_obs[name]["current_lidar"]

        prev_throttle = np.array([self.multi_agent_control[name].throttle])
        prev_steering = np.array([self.multi_agent_control[name].steering])

        if lidar_error:
            observation = {  # dummy observation, the sim will end anyway
                "current_lidar": current_lidar,
                "prev_lidar1": prev_lidar1,
                "prev_lidar2": prev_lidar2,
                "prev_lidar3": prev_lidar3,
                "prev_lidar4": prev_lidar4,
                "prev_lidar5": prev_lidar5,
                "prev_throttle": prev_throttle,
                "prev_steering": prev_steering,
            }

            print(
                """"Caution, no point was observed by the lidar, the vehicule may be escaping:
                    reseting sim"""
            )
            # Alas, Done cannot be returned by init, but step() will take care of ending the sim
            self.done = True

        if self.reversed_world:
            current_lidar[:, 0] *= -1
            current_lidar = current_lidar[::-1]
            prev_lidar1[:, 0] *= -1
            prev_lidar1 = prev_lidar1[::-1]
            prev_lidar2[:, 0] *= -1
            prev_lidar2 = prev_lidar2[::-1]
            prev_lidar3[:, 0] *= -1
            prev_lidar3 = prev_lidar3[::-1]
            prev_lidar4[:, 0] *= -1
            prev_lidar4 = prev_lidar4[::-1]
            prev_lidar5[:, 0] *= -1
            prev_lidar5 = prev_lidar5[::-1]

        observation = {
            "current_lidar": current_lidar,
            "prev_lidar1": prev_lidar1,
            "prev_lidar2": prev_lidar2,
            "prev_lidar3": prev_lidar3,
            "prev_lidar4": prev_lidar4,
            "prev_lidar5": prev_lidar5,
            "prev_throttle": prev_throttle,
            "prev_steering": prev_steering,
        }
        self.multi_agent_obs[name] = observation


class MultiDiscreteAirSimEnv(BaseAirSimEnv):
    """Custom AirSim Environment MultiDiscrete action space that follows gym interface."""

    def __init__(
        self,
        client,
        lidar_size,
        dt,
        ClockSpeed,
        UE_spawn_point,
        liste_checkpoints_coordinates,
        liste_spawn_point,
        is_rendered=False,
        random_reverse=False,
    ):
        """


        Parameters
        ----------
        client : TYPE
            AirSim client.
        lidar_size : int
            Number of point observed by the lidar in each observation.
        dt : float
            Simulation time step separating two observations/actions.
        ClockSpeed : int
            ClockSpeed selected in SETTINGS.JSON. Make sure that FPS/ClockSpeed > 30
        UE_spawn_point : numpy array
            coordinates of the "player spawn" object in unreal engine
        liste_checkpoints_coordinates : list [x,y,r]
            list of the coordinates of the checkpoints. In UE metric system.
        is_rendered : Boolean, optional
            Wether or not the lidar map is rendering. The default is False.
        random_reverse : boolean
            whether or not the the observations and action should randomly be reversed.
            Strongly advised for training


        Returns
        -------
        None.

        """
        super().__init__(
            client,
            lidar_size,
            dt,
            ClockSpeed,
            UE_spawn_point,
            liste_checkpoints_coordinates,
            liste_spawn_point,
            is_rendered=False,
            random_reverse=False,
        )

        # _____________ Below are MDP related objects _______________________

        self.action_space = gym.spaces.MultiDiscrete([3, 5])

        # In this order
        # "throttle" -> 0 = forward  ; 1 = stop        ; 2 = reverse
        # "steering" -> 0 = full left; 1 = middle left ; 2 = straight;
        #                            ; 3 = middle right; 4 = full right

        self.throttle_arg_to_throttle = {0: 1, 1: 0, 2: -1}
        self.steering_arg_to_steering = {0: -0.5, 1: -0.25, 2: 0, 3: 0.25, 4: 0.5}

        # Example for using image as input (channel-first; channel-last also works):

        low = np.zeros((lidar_size, 2), dtype=np.float32)
        high = np.zeros((lidar_size, 2), dtype=np.float32)

        low[:, 0] = -np.pi
        low[:, 1] = 0

        high[:, 0] = np.pi
        high[:, 1] = np.inf

        self.observation_space = gym.spaces.Dict(
            spaces={
                # the format is [angle , radius]
                "current_lidar": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_lidar": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_throttle": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "prev_steering": gym.spaces.Box(low=-0.5, high=+0.5, shape=(1,)),
            }
        )

    def step(self, action):
        info = {}  # Just a debugging feature here

        # The actions are extracted from the "action" argument
        throttle_arg = int(action[0])
        steering_arg = int(action[1])

        self.car_controls.throttle = self.throttle_arg_to_throttle[throttle_arg]
        self.car_controls.steering = self.steering_arg_to_steering[steering_arg]

        if self.reversed_world:
            self.car_controls.steering *= -1

        self.client.setCarControls(self.car_controls)

        # Now that everything is good and proper, let's run  AirSim a bit
        self.airsim_step()

        # _______ updating the checkpoint situation ___________________

        self.checkpoint_update()

        # __________ extracts the observation ______________________

        self.observation_maker(init=False)

        # ___________ Updates the reward ____________________

        self.reward_calculation()

        # displays ( or not ) the lidar observation
        if self.is_rendered:
            self.render()

        return self.observation, self.reward, self.done, info

    def reset(self):

        self.client.reset()

        # _______ random respawn ________________________

        self.random_respawn()

        # ______ Init agent ______________________________

        self.car_controls.throttle = 0
        self.car_controls.steering = 0

        self.init_values()

        self.observation_maker(init=True)

        return self.observation  # reward, done, info can't be included

    def observation_maker(self, init=False):
        """ Fetch observation

        Contains all sort of operation on lidar data. Proximity jammer, uniform
        sampling, and a killswitch if no lidar point is detected.
        Parameters
        ----------
        init : Boolean, optional
            Whether or not this observation is called by reset() or step().
            Basically, whether or not the lidar memory is empty.
            The default is False.
        Returns
        -------
        None.
        """

        current_raw_lidar = convert_lidar_data_to_polar(self.client.getLidarData())
        current_raw_lidar = proximity_jammer(
            current_raw_lidar
        )  # removing closest points
        current_lidar, lidar_error = lidar_formater(current_raw_lidar, self.lidar_size)

        if init:  # Initialising memory
            prev_lidar = np.copy(current_lidar)
        else:
            prev_lidar = self.observation["current_lidar"]

        prev_throttle = np.array([self.car_controls.throttle])
        prev_steering = np.array([self.car_controls.steering])

        if lidar_error:
            print(
                """"Caution, no point was observed by the lidar, the vehicule may be escaping:
                    reseting sim"""
            )
            # Alas, Done cannot be returned by init, but step() will take care of ending the sim
            self.done = True

        if self.reversed_world:
            current_lidar[:, 0] *= -1
            current_lidar = current_lidar[::-1]
            prev_lidar[:, 0] *= -1
            prev_lidar = prev_lidar[::-1]

        observation = {
            "current_lidar": current_lidar,
            "prev_lidar": prev_lidar,
            "prev_throttle": prev_throttle,
            "prev_steering": prev_steering,
        }
        self.observation = observation


class DiscreteAirSimEnv(BaseAirSimEnv):
    """Custom AirSim Environment Discrete action space that follows gym interface"""

    def __init__(
        self,
        client,
        lidar_size,
        dt,
        ClockSpeed,
        UE_spawn_point,
        liste_checkpoints_coordinates,
        liste_spawn_point,
        is_rendered=False,
        random_reverse=False,
    ):
        """


        Parameters
        ----------
        client : TYPE
            AirSim client.
        lidar_size : int
            Number of point observed by the lidar in each observation.
        dt : float
            Simulation time step separating two observations/actions.
        ClockSpeed : int
            ClockSpeed selected in SETTINGS.JSON. Make sure that FPS/ClockSpeed > 30
        UE_spawn_point : numpy array
            coordinates of the "player spawn" object in unreal engine
        liste_checkpoints_coordinates : list [x,y,r]
            list of the coordinates of the checkpoints. In UE metric system.
        is_rendered : Boolean, optional
            Whether or not the lidar map is rendering. The default is False.
        random_reverse : boolean
            whether or not the the observations and action should randomly be reversed.
            Strongly advised for training


        Returns
        -------
        None.

        """
        super().__init__(
            client,
            lidar_size,
            dt,
            ClockSpeed,
            UE_spawn_point,
            liste_checkpoints_coordinates,
            liste_spawn_point,
            is_rendered=False,
            random_reverse=False,
        )

        # ___________ Below are MDP related objects _____________________________

        self.action_space = gym.spaces.Discrete(5)

        # In this order
        # "steering" -> 0 = full left; 1 = middle left ; 2 = straight;
        #                            ; 3 = middle right; 4 = full right

        self.steering_arg_to_steering = {0: -0.5, 1: -0.25, 2: 0, 3: 0.25, 4: 0.5}

        low = np.zeros((lidar_size, 2), dtype=np.float32)
        high = np.zeros((lidar_size, 2), dtype=np.float32)

        low[:, 0] = -np.pi
        low[:, 1] = 0

        high[:, 0] = np.pi
        high[:, 1] = np.inf

        self.observation_space = gym.spaces.Dict(
            spaces={
                # the format is [angle , radius]
                "current_lidar": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_lidar": gym.spaces.Box(
                    low=low, high=high, shape=(lidar_size, 2), dtype=np.float32
                ),
                "prev_steering": gym.spaces.Box(low=-0.5, high=+0.5, shape=(1,)),
            }
        )

    def step(self, action):
        info = {}  # Just a debugging feature here

        # The actions are extracted from the "action" argument
        steering_arg = int(action)

        self.car_controls.throttle = 0.5
        self.car_controls.steering = self.steering_arg_to_steering[steering_arg]

        if self.reversed_world:
            self.car_controls.steering *= -1

        self.client.setCarControls(self.car_controls)

        # Now that everything is good and proper, let's run  AirSim a bit
        self.airsim_step()

        # _______ updating the checkpoint situation ___________________

        self.checkpoint_update()

        # __________ extracts the observation ______________________

        self.observation_maker(init=False)

        # ___________ Updates the reward ____________________

        self.reward_calculation()

        # displays ( or not ) the lidar observation
        if self.is_rendered:
            self.render()

        return self.observation, self.reward, self.done, info

    def reset(self):

        self.client.reset()

        # _______ random respawn ________________________

        self.random_respawn()

        # ______ Init agent ______________________________

        self.car_controls.throttle = 0.8
        self.car_controls.steering = 0

        self.init_values()

        self.observation_maker(init=True)

        return self.observation  # reward, done, info can't be included

    def observation_maker(self, init=False):
        """ Fetch observation

        Contains all sort of operation on lidar data. Proximity jammer, uniform
        sampling, and a killswitch if no lidar point is detected.
        Parameters
        ----------
        init : Boolean, optional
            Whether or not this observation is called by reset() or step().
            Basically, whether or not the lidar memory is empty.
            The default is False.
        Returns
        -------
        None.
        """

        current_raw_lidar = convert_lidar_data_to_polar(self.client.getLidarData())
        current_raw_lidar = proximity_jammer(
            current_raw_lidar
        )  # removing closest points
        current_lidar, lidar_error = lidar_formater(current_raw_lidar, self.lidar_size)

        if init:  # Initialising memory
            prev_lidar = np.copy(current_lidar)
        else:
            prev_lidar = self.observation["current_lidar"]

        prev_steering = np.array([self.car_controls.steering])

        if lidar_error:
            print(
                """"Caution, no point was observed by the lidar, the vehicule may be escaping:
                    reseting sim"""
            )
            # Alas, Done cannot be returned by init, but step() will take care of ending the sim
            self.done = True

        if self.reversed_world:
            current_lidar[:, 0] *= -1
            current_lidar = current_lidar[::-1]
            prev_lidar[:, 0] *= -1
            prev_lidar = prev_lidar[::-1]

        observation = {
            "current_lidar": current_lidar,
            "prev_lidar": prev_lidar,
            "prev_steering": prev_steering,
        }
        self.observation = observation
