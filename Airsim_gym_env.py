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


class BoxAirSimEnv(gym.Env):
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
        super(BoxAirSimEnv, self).__init__()

        # __________ Init value __________________
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

        # ________ init of reset() attributes ___________

        self.reversed_world = None
        self.prev_throttle = None
        self.prev_steering = None
        self.current_lidar = None
        self.prev_lidar = None
        self.Circuit1 = None
        self.throttle = None
        self.steering = None
        self.ax = None

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
        self.client.simPause(False)
        # TODO a dedicated thread may be more efficient
        time.sleep(self.dt / self.ClockSpeed)
        self.client.simPause(True)

        # Get the state from AirSim
        self.car_state = self.client.getCarState()

        self.prev_throttle = np.array([action[0]])
        self.prev_steering = np.array([action[1]])

        position = self.car_state.kinematics_estimated.position
        gate_passed = self.Circuit1.cycle_tick(position.x_val, position.y_val)[
            0
        ]  # updating the checkpoint situation

        # __________ extracts the observation _____________________________________
        current_raw_lidar = convert_lidar_data_to_polar(self.client.getLidarData())

        self.current_lidar, lidar_error = lidar_formater(
            current_raw_lidar, target_lidar_size=self.lidar_size
        )
        # ______________ Error killswitch, in case the car is going rogue ! ______________
        if (
            lidar_error or self.done
        ):  # self.done can never be true at this point unless the lidar was corrupted in reset()
            observation = {  # dummy observation, save the sim !
                "current_lidar": self.current_lidar,
                "prev_lidar": self.prev_lidar,
                "prev_throttle": self.prev_throttle,
                "prev_steering": self.prev_steering,
            }
            print(
                """Caution, no point was observed by the lidar, the vehicule may be escaping:
                    reseting sim"""
            )
            return observation, 0, True, {"Lidar error": True}

        # __________ Reversing the world for data augmentation ________________
        if self.reversed_world:
            self.current_lidar[:, 0] *= -1
            self.current_lidar = self.current_lidar[::-1]
        # ____________ Observation __________________
        observation = {
            "current_lidar": self.current_lidar,
            "prev_lidar": self.prev_lidar,
            "prev_throttle": self.prev_throttle,
            "prev_steering": self.prev_steering,
        }

        self.prev_lidar = self.current_lidar

        # ________________ Updates the reward ________________
        # collision info is necessary to compute reward
        collision_info = self.client.simGetCollisionInfo()
        crash = collision_info.has_collided

        reward = 0
        if crash:
            reward = -100

        elif self.car_state.speed <= 1:  # Lets force the car to move
            reward = -0.1

        if gate_passed:
            reward += 50
            print("gate_passed")

        self.total_reward = self.total_reward + reward

        if crash:
            self.done = True
            print("Crash occured")
            print("Episode reward : " + str(self.total_reward) + 2 * "\n")

        # displays ( or not ) the lidar observation
        if self.is_rendered:
            self.render()

        return observation, reward, self.done, info

    def reset(self):

        self.client.reset()

        # Picking a random spawn
        Circuit_wrapper1 = Circuit_wrapper(
            self.liste_spawn_point,
            self.liste_checkpoints_coordonnes,
            UE_spawn_point=self.UE_spawn_point,
        )
        spawn_point, theta, self.Circuit1 = Circuit_wrapper1.sample_random_spawn_point()

        # be careful, the call arguments for quaterninons are x,y,z,w
        x_val, y_val, z_val = spawn_point.x, spawn_point.y, spawn_point.z

        pose = airsim.Pose()

        orientation = airsim.Quaternionr(0, 0, np.sin(theta / 2) * 1, np.cos(theta / 2))
        position = airsim.Vector3r(x_val, y_val, z_val)
        pose.position = position
        pose.orientation = orientation
        self.client.simSetVehiclePose(pose, ignore_collision=True)

        ##########

        self.throttle = 0
        self.steering = 0

        self.done = False

        self.total_reward = 0
        # let's skip the first frames to inialise lidar and make sure everything is right
        self.client.simContinueForFrames(100)
        # the lidar data can take a bit of time before initialisation.
        time.sleep(1)

        current_raw_lidar = convert_lidar_data_to_polar(self.client.getLidarData())
        self.current_lidar, lidar_error = lidar_formater(
            current_raw_lidar, self.lidar_size
        )
        self.prev_lidar = np.copy(self.current_lidar)

        print("reset")
        init_normalized_action = normalize_action(np.array([0, 0]))
        self.throttle = init_normalized_action[0]
        self.steering = init_normalized_action[1]
        self.prev_throttle = np.array([init_normalized_action[0]])
        self.prev_steering = np.array([init_normalized_action[1]])

        if lidar_error:
            observation = {  # dummy observation, the sim will end anyway
                "current_lidar": self.current_lidar,
                "prev_lidar": self.prev_lidar,
                "prev_throttle": self.prev_throttle,
                "prev_steering": self.prev_steering,
            }

            print(
                """"Caution, no point was observed by the lidar, the vehicule may be escaping:
                    reseting sim"""
            )
            # Alas, Done cannot be returned by init, but step() will take care of ending the sim
            self.done = True

        if self.random_reverse:
            self.reversed_world = random.choice([False, True])

        if self.reversed_world:
            self.current_lidar[:, 0] *= -1
            self.current_lidar = self.current_lidar[::-1]
            self.prev_lidar[:, 0] *= -1
            self.prev_lidar = self.current_lidar[::-1]

        observation = {
            "current_lidar": self.current_lidar,
            "prev_lidar": self.prev_lidar,
            "prev_throttle": self.prev_throttle,
            "prev_steering": self.prev_steering,
        }

        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        if not self.is_rendered:
            fig = plt.figure()
            self.ax = fig.add_subplot(projection="polar")
            self.is_rendered = True

        self.ax.clear()
        T = self.current_lidar[:, 0]
        R = self.current_lidar[:, 1]
        self.ax.scatter(T, R)
        plt.pause(0.01)
        plt.draw()

        # ____________ Image ___________________
        responses = self.client.simGetImages(
            [airsim.ImageRequest("Camera1", airsim.ImageType.Scene, False, False)],
            "A_MyVehicle",
        )  # scene vision image in uncompressed RGB array
        response = responses[0]

        # get numpy array
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)

        cv2.imshow("image", img_rgb)

    def close(self):
        self.client.simPause(False)
        self.client.enableApiControl(False)

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


class BoxAirSimEnv_5_memory(gym.Env):
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
        super(BoxAirSimEnv_5_memory, self).__init__()

        # Define action and observation space

        self.client = client
        self.multi_agent_control = {}
        self.multi_agent_control["A_MyVehicle"] = airsim.CarControls()
        self.ennemi_number = 4
        for i in range(1, self.ennemi_number + 1):
            self.multi_agent_control["Car{}".format(i)] = airsim.CarControls()
            self.multi_agent_control["Car{}".format(i)].throttle = 0
            self.multi_agent_control["Car{}".format(i)].steering = 0

        self.car_state = client.getCarState("A_MyVehicle")
        # car_state is an AirSim object that contains informations that are not
        # obtainable in real experiments. Therefore, it cannot be used as a
        # MDP object. However, it is useful for computing the reward

        self.is_rendered = is_rendered
        self.dt = dt
        self.ClockSpeed = ClockSpeed
        self.lidar_size = lidar_size
        self.random_reverse = random_reverse
        self.reversed_world = None

        self.total_reward = 0
        self.done = False

        self.UE_spawn_point = UE_spawn_point
        self.liste_checkpoints_coordonnes = liste_checkpoints_coordinates
        self.liste_spawn_point = liste_spawn_point

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

        # ________ init of reset() attributes ___________

        self.reversed_world = None
        self.multi_agent_obs = {}
        self.Circuit1 = None
        self.ax = None

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
        self.client.simPause(False)
        # TODO a dedicated thread may be more efficient
        time.sleep(self.dt / self.ClockSpeed)
        self.client.simPause(True)

        # Get the state from AirSim
        self.car_state = self.client.getCarState("A_MyVehicle")
        for i in range(5):
            self.observation_maker(i)

        position = self.car_state.kinematics_estimated.position
        gate_passed = self.Circuit1.cycle_tick(position.x_val, position.y_val)[
            0
        ]  # updating the checkpoint situation

        # ___________ Updates the reward ____________________
        # collision info is necessary to compute reward
        collision_info = self.client.simGetCollisionInfo("A_MyVehicle")
        crash = collision_info.has_collided

        reward = 0
        if crash:
            reward = -100

        elif self.car_state.speed <= 1:  # Lets force the car to move
            reward = -0.1

        if gate_passed:
            reward += 50
            print("gate_passed")

        self.total_reward = self.total_reward + reward

        if crash:
            self.done = True
            print("Crash occured")
            print("Episode reward : " + str(self.total_reward) + 2 * "\n")

        # displays ( or not ) the lidar observation
        if self.is_rendered:
            self.render()

        return self.multi_agent_obs["A_MyVehicle"], reward, self.done, info

    def reset(self):

        print("reset")
        self.client.reset()

        # be careful, the call arguments for quaterninons are x,y,z,w

        # _______ main car respawn ______________________________

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
        self.client.simSetVehiclePose(
            pose, ignore_collision=True, vehicle_name="A_MyVehicle"
        )

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

        ##########

        self.done = False

        self.total_reward = 0
        # let's skip the first frames to inialise lidar and make sure everything is right
        self.client.simContinueForFrames(100)
        # the lidar data can take a bit of time before initialisation.
        time.sleep(1)

        if self.random_reverse:
            self.reversed_world = random.choice([False, True])

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
        if not (1 <= i <= self.ennemi_number):
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

    def render(self, mode="human"):
        if not self.is_rendered:
            fig = plt.figure()
            self.ax = fig.add_subplot(projection="polar")
            self.is_rendered = True

        self.ax.clear()
        T = self.current_lidar[:, 0]
        R = self.current_lidar[:, 1]
        self.ax.scatter(T, R)
        plt.pause(1e-6)
        plt.draw()

    # =============================================================================
    #         # ________________ Image ___________________________________________________
    #         responses = self.client.simGetImages(
    #             [airsim.ImageRequest("Camera1", airsim.ImageType.Scene, False, False)],
    #             "A_MyVehicle",
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

    def close(self):
        self.client.simPause(False)
        self.client.enableApiControl(False)

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


class MultiDiscreteAirSimEnv(gym.Env):
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
        super(MultiDiscreteAirSimEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

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

        self.total_reward = 0
        self.done = False

        self.random_reverse = random_reverse
        self.reversed_world = None

        self.UE_spawn_point = UE_spawn_point
        self.liste_checkpoints_coordonnes = liste_checkpoints_coordinates
        self.liste_spawn_point = liste_spawn_point

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

        # ________ init of reset() attributes ___________

        self.reversed_world = None
        self.prev_throttle = None
        self.prev_steering = None
        self.current_lidar = None
        self.prev_lidar = None
        self.Circuit1 = None
        self.throttle = None
        self.steering = None
        self.ax = None

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
        self.client.simPause(False)
        # TODO a dedicated thread may be more efficient
        time.sleep(self.dt / self.ClockSpeed)
        self.client.simPause(True)

        # Get the state from AirSim
        self.car_state = self.client.getCarState()

        self.prev_throttle = np.array([self.car_controls.throttle])
        self.prev_steering = np.array([self.car_controls.steering])

        position = self.car_state.kinematics_estimated.position
        gate_passed = self.Circuit1.cycle_tick(position.x_val, position.y_val)[
            0
        ]  # updating the checkpoint situation

        # _______________ extracts the observation ____________________
        current_raw_lidar = convert_lidar_data_to_polar(self.client.getLidarData())

        self.current_lidar, lidar_error = lidar_formater(
            current_raw_lidar, target_lidar_size=self.lidar_size
        )
        # _____________ Error killswitch, in case the car is going rogue ! ________________
        if (
            lidar_error or self.done
        ):  # self.done can never be true at this point unless the lidar was corrupted in reset()
            observation = {  # dummy observation, save the sim !
                "current_lidar": self.current_lidar,
                "prev_lidar": self.prev_lidar,
                "prev_throttle": self.prev_throttle,
                "prev_steering": self.prev_steering,
            }
            print(
                """"Caution, no point was observed by the lidar, the vehicule may be escaping:
                    reseting sim"""
            )
            return observation, 0, True, {"Lidar error": True}

        # __________ Reversing the world for data augmentation ________________
        if self.reversed_world:
            self.current_lidar[:, 0] *= -1
            self.current_lidar = self.current_lidar[::-1]
        # ___________ Observation _____________________________
        observation = {
            "current_lidar": self.current_lidar,
            "prev_lidar": self.prev_lidar,
            "prev_throttle": self.prev_throttle,
            "prev_steering": self.prev_steering,
        }

        self.prev_lidar = self.current_lidar

        # _______ Updates the reward ______________________________
        # collision info is necessary to compute reward
        collision_info = self.client.simGetCollisionInfo()
        crash = collision_info.has_collided

        reward = 0
        if crash:
            reward = -100

        elif self.car_state.speed <= 1:  # Lets force the car to move
            reward = -0.1

        if gate_passed:
            reward += 50
            print("gate_passed")

        self.total_reward = self.total_reward + reward

        if crash:
            self.done = True
            print("Crash occured")
            print("Episode reward : " + str(self.total_reward) + 2 * "\n")

        # displays ( or not ) the lidar observation
        if self.is_rendered:
            self.render()

        return observation, reward, self.done, info

    def reset(self):

        self.client.reset()

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
        self.client.simSetVehiclePose(pose, ignore_collision=True)

        # ____________________________________________

        self.throttle = 0
        self.steering = 0

        self.done = False

        self.total_reward = 0
        # let's skip the first frames to inialise lidar and make sure everything is right
        self.client.simContinueForFrames(100)
        # the lidar data can take a bit of time before initialisation.
        time.sleep(1)

        current_raw_lidar = convert_lidar_data_to_polar(self.client.getLidarData())
        self.current_lidar, lidar_error = lidar_formater(
            current_raw_lidar, self.lidar_size
        )
        self.prev_lidar = np.copy(self.current_lidar)

        print("reset")

        self.prev_throttle = np.array([0])
        self.prev_steering = np.array([0])

        if lidar_error:
            observation = {  # dummy observation, the sim will end anyway
                "current_lidar": self.current_lidar,
                "prev_lidar": self.prev_lidar,
                "prev_throttle": self.prev_throttle,
                "prev_steering": self.prev_steering,
            }

            print(
                """"Caution, no point was observed by the lidar, the vehicule may be escaping:
                    reseting sim"""
            )
            # Alas, Done cannot be returned by init, but step() will take care of ending the sim
            self.done = True

        if self.random_reverse:
            self.reversed_world = random.choice([False, True])

        if self.reversed_world:
            self.current_lidar[:, 0] *= -1
            self.current_lidar = self.current_lidar[::-1]
            self.prev_lidar[:, 0] *= -1
            self.prev_lidar = self.current_lidar[::-1]

        observation = {
            "current_lidar": self.current_lidar,
            "prev_lidar": self.prev_lidar,
            "prev_throttle": self.prev_throttle,
            "prev_steering": self.prev_steering,
        }

        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        if not self.is_rendered:
            fig = plt.figure()
            self.ax = fig.add_subplot(projection="polar")
            self.is_rendered = True

        self.ax.clear()
        T = self.current_lidar[:, 0]
        R = self.current_lidar[:, 1]
        self.ax.scatter(T, R)
        plt.pause(0.01)
        plt.draw()

        # ______________ Image _______________________
        responses = self.client.simGetImages(
            [airsim.ImageRequest("Camera1", airsim.ImageType.Scene, False, False)],
            "A_MyVehicle",
        )  # scene vision image in uncompressed RGB array
        response = responses[0]

        # get numpy array
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)

        cv2.imshow("image", img_rgb)

    def close(self):
        self.client.simPause(False)
        self.client.enableApiControl(False)

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


class DiscreteAirSimEnv(gym.Env):
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
        super(DiscreteAirSimEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

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
        self.reversed_world = None

        self.total_reward = 0
        self.done = False
        self.step_iterations = 0

        self.UE_spawn_point = UE_spawn_point
        self.liste_checkpoints_coordonnes = liste_checkpoints_coordinates
        self.liste_spawn_point = liste_spawn_point

        # ___________ Below are MDP related objects _____________________________

        self.action_space = gym.spaces.Discrete(5)

        # In this order
        # "steering" -> 0 = full left; 1 = middle left ; 2 = straight;
        #                            ; 3 = middle right; 4 = full right

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
                "prev_steering": gym.spaces.Box(low=-0.5, high=+0.5, shape=(1,)),
            }
        )
        # ________ init of reset() attributes ___________

        self.reversed_world = None
        self.prev_steering = None
        self.current_lidar = None
        self.prev_lidar = None
        self.Circuit1 = None
        self.steering = None
        self.ax = None

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
        self.client.simPause(False)
        # TODO a dedicated thread may be more efficient
        time.sleep(self.dt / self.ClockSpeed)
        self.client.simPause(True)

        # Get the state from AirSim
        self.car_state = self.client.getCarState()

        self.prev_steering = np.array([self.car_controls.steering])

        position = self.car_state.kinematics_estimated.position
        gate_passed = self.Circuit1.cycle_tick(position.x_val, position.y_val)[
            0
        ]  # updating the checkpoint situation

        # __________________ extracts the observation __________________________
        current_raw_lidar = convert_lidar_data_to_polar(self.client.getLidarData())

        self.current_lidar, lidar_error = lidar_formater(
            current_raw_lidar, target_lidar_size=self.lidar_size
        )
        # _____________ Error killswitch, in case the car is going rogue ! __________________
        if (
            lidar_error or self.done
        ):  # self.done can never be true at this point unless the lidar was corrupted in reset()
            observation = {  # dummy observation, save the sim !
                "current_lidar": self.current_lidar,
                "prev_lidar": self.prev_lidar,
                "prev_steering": self.prev_steering,
            }
            print(
                """"Caution, no point was observed by the lidar, the vehicule may be escaping:
                    reseting sim"""
            )
            return observation, 0, True, {"Lidar error": True}

        # ___________ Reversing the world for data augmentation _________________
        if self.reversed_world:
            self.current_lidar[:, 0] *= -1
            self.current_lidar = self.current_lidar[::-1]
        # _________ Observation ________________________________
        observation = {
            "current_lidar": self.current_lidar,
            "prev_lidar": self.prev_lidar,
            "prev_steering": self.prev_steering,
        }

        self.prev_lidar = self.current_lidar

        # _________ Updates the reward __________________________
        # collision info is necessary to compute reward
        collision_info = self.client.simGetCollisionInfo()
        crash = collision_info.has_collided

        reward = 0
        if crash:
            reward = -100

        elif self.car_state.speed <= 1:  # Lets force the car to move
            reward = -0.1

        if gate_passed:
            reward += 50
            print("gate_passed")

        self.total_reward = self.total_reward + reward

        if crash:
            self.done = True
            print("Crash occured")
            print("Episode reward : " + str(self.total_reward) + 2 * "\n")

        # displays ( or not ) the lidar observation
        if self.is_rendered:
            self.render()

        return observation, reward, self.done, info

    def reset(self):

        self.client.reset()

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
        self.client.simSetVehiclePose(pose, ignore_collision=True)

        ##########

        self.steering = 0

        self.done = False

        self.total_reward = 0
        # let's skip the first frames to inialise lidar and make sure everything is right
        self.client.simContinueForFrames(100)
        # the lidar data can take a bit of time before initialisation.
        time.sleep(1)

        current_raw_lidar = convert_lidar_data_to_polar(self.client.getLidarData())
        self.current_lidar, lidar_error = lidar_formater(
            current_raw_lidar, self.lidar_size
        )
        self.prev_lidar = np.copy(self.current_lidar)

        print("reset")

        self.prev_steering = np.array([0])

        if lidar_error:
            observation = {  # dummy observation, the sim will end anyway
                "current_lidar": self.current_lidar,
                "prev_lidar": self.prev_lidar,
                "prev_steering": self.prev_steering,
            }

            print(
                """"Caution, no point was observed by the lidar, the vehicule may be escaping:
                    reseting sim"""
            )
            # Alas, Done cannot be returned by init, but step() will take care of ending the sim
            self.done = True

        if self.random_reverse:
            self.reversed_world = random.choice([False, True])

        if self.reversed_world:
            self.current_lidar *= -1
            self.current_lidar = self.current_lidar[::-1]
            self.prev_lidar *= -1
            self.prev_lidar = self.current_lidar[::-1]

        observation = {
            "current_lidar": self.current_lidar,
            "prev_lidar": self.prev_lidar,
            "prev_steering": self.prev_steering,
        }

        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        if not self.is_rendered:
            fig = plt.figure()
            self.ax = fig.add_subplot(projection="polar")
            self.is_rendered = True

        self.ax.clear()
        T = self.current_lidar[:, 0]
        R = self.current_lidar[:, 1]
        self.ax.scatter(T, R)
        plt.pause(0.01)
        plt.draw()

        # ___________ Image _________________________________
        responses = self.client.simGetImages(
            [airsim.ImageRequest("Camera1", airsim.ImageType.Scene, False, False)],
            "A_MyVehicle",
        )  # scene vision image in uncompressed RGB array
        response = responses[0]

        # get numpy array
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)

        cv2.imshow("image", img_rgb)

    def close(self):
        self.client.simPause(False)
        self.client.enableApiControl(False)

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
