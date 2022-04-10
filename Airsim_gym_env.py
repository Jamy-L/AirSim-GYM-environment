# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:05:35 2022

@author: jamyl
"""


import gym
import airsim
import numpy as np
import time
from jamys_toolkit import Circuit_wrapper, convert_lidar_data_to_polar
import matplotlib.pyplot as plt
import cv2
import random
from threading import Thread


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


def sparse_sample(X, sample_size):
    """Uniform sampling of the X array.

    Sorting x beforehand may be a good idea.

    Parameters
    ----------
    X : numpy array
        Lidar data to which sample uniformally
    sample_size : int
         The desired output size

    Returns
    -------
    numpy array
        A smaller version of X

    """
    N = X.shape[0]
    if N < sample_size:
        raise ValueError(
            """"The lidar to sample is smaller than the wanted sample_size
            (size {} against target size {}). Maybe you should use
            array_augmentation() instead""".format(
                N, sample_size
            )
        )

    Indexes = np.linspace(0, N - 1, sample_size)
    return X[Indexes.astype(np.int32)]


def array_augmentation(A, final_size):
    """Transform of A into a bigger array.

    Parameters
    ----------
    A : numpy array
        Lidar data
    final_size : int
        The desired finals size

    Returns
    -------
    B : numpy array
        conversion of A in a bigger size

    """
    N = A.shape[0]
    if N > final_size:
        raise ValueError(
            """"The lidar data to augmentate is bigger than the target size
            (size {} against target size {}). Maybe you should use
            sparse_sample() instead""".format(
                N, final_size
            )
        )
    m = final_size - N
    B1 = np.ones((m, 2)) * A[0]
    B = np.concatenate((B1, A[0:, :]), axis=0)
    return B


def lidar_formater(lidar_data, target_lidar_size, angle_sort=True):
    """
    transforms a variable size lidar data to a fixed size numpy array. If the
    lidar is too big, points are cropped. If it's too small, the first value
    is padded. If lidar_data is empty, it is filled with zeros and
    conversion_error = True is returned


    Parameters
    ----------
    lidar_data : numpy array
        Numy array reprensenting the polar converted raw lidar data received.
    target_lidar_size : int
        Number of points desired for output
    angle_srot : boolean
        Whether or not the values should be ordered by growing theta (may
        impact performances if the target lidar size is very big)

    Returns
    -------
    new_lidar_data : numpy array
        size adjusted new lidar data
    conversion_error: boolean
        whether or not an error occured

    """

    if angle_sort:
        idx = lidar_data[:, 0].argsort()
        lidar_data = lidar_data[idx, :]

    n_points_received = lidar_data.shape[0]
    if lidar_data.size == 0:
        return np.zeros((target_lidar_size, 2)), True

    if n_points_received < target_lidar_size:  # not enough points !
        new_lidar_data = array_augmentation(lidar_data, target_lidar_size)

    else:
        new_lidar_data = sparse_sample(lidar_data, target_lidar_size)

    return new_lidar_data, False


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
            "MyVehicle",
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

        self.UE_spawn_point = UE_spawn_point
        self.liste_checkpoints_coordonnes = liste_checkpoints_coordinates
        self.liste_spawn_point = liste_spawn_point

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
        self.prev_throttle = None
        self.prev_steering = None
        self.current_lidar = None
        self.prev_lidar1 = None
        self.prev_lidar2 = None
        self.prev_lidar3 = None
        self.prev_lidar4 = None
        self.prev_lidar5 = None
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

        # __________ extracts the observation ______________________
        current_raw_lidar = convert_lidar_data_to_polar(self.client.getLidarData())
        current_raw_lidar = proximity_jammer(
            current_raw_lidar
        )  # removing closest points
        self.current_lidar, lidar_error = lidar_formater(
            current_raw_lidar, target_lidar_size=self.lidar_size
        )

        # __________ Error killswitch, in case the car is going rogue ! _____________
        if (
            lidar_error or self.done
        ):  # self.done can never be true at this point unless the lidar was corrupted in reset()
            observation = {  # dummy observation, save the sim !
                "current_lidar": self.current_lidar,
                "prev_lidar1": self.prev_lidar1,
                "prev_lidar2": self.prev_lidar2,
                "prev_lidar3": self.prev_lidar3,
                "prev_lidar4": self.prev_lidar4,
                "prev_lidar5": self.prev_lidar5,
                "prev_throttle": self.prev_throttle,
                "prev_steering": self.prev_steering,
            }
            print(
                """"Caution, no point was observed by the lidar, the vehicule may be escaping:
                    reseting sim"""
            )
            return observation, 0, True, {"Lidar error": True}

        # ______________________________________________

        if self.reversed_world:
            self.current_lidar[:, 0] *= -1
            self.current_lidar = self.current_lidar[::-1]
            self.prev_lidar1[:, 0] *= -1
            self.prev_lidar1 = self.prev_lidar1[::-1]
            self.prev_lidar2[:, 0] *= -1
            self.prev_lidar2 = self.prev_lidar2[::-1]
            self.prev_lidar3[:, 0] *= -1
            self.prev_lidar3 = self.prev_lidar3[::-1]
            self.prev_lidar4[:, 0] *= -1
            self.prev_lidar4 = self.prev_lidar4[::-1]
            self.prev_lidar5[:, 0] *= -1
            self.prev_lidar5 = self.prev_lidar5[::-1]

        observation = {
            "current_lidar": self.current_lidar,
            "prev_lidar1": self.prev_lidar1,
            "prev_lidar2": self.prev_lidar2,
            "prev_lidar3": self.prev_lidar3,
            "prev_lidar4": self.prev_lidar4,
            "prev_lidar5": self.prev_lidar5,
            "prev_throttle": self.prev_throttle,
            "prev_steering": self.prev_steering,
        }
        self.prev_lidar5 = self.prev_lidar4
        self.prev_lidar4 = self.prev_lidar3
        self.prev_lidar3 = self.prev_lidar2
        self.prev_lidar2 = self.prev_lidar1
        self.prev_lidar1 = self.current_lidar

        # ___________ Updates the reward ____________________
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

        self.prev_lidar1 = np.copy(self.current_lidar)
        self.prev_lidar2 = np.copy(self.current_lidar)
        self.prev_lidar3 = np.copy(self.current_lidar)
        self.prev_lidar4 = np.copy(self.current_lidar)
        self.prev_lidar5 = np.copy(self.current_lidar)

        print("reset")
        init_normalized_action = normalize_action(np.array([0, 0]))
        self.throttle = init_normalized_action[0]
        self.steering = init_normalized_action[1]
        self.prev_throttle = np.array([init_normalized_action[0]])
        self.prev_steering = np.array([init_normalized_action[1]])

        if lidar_error:
            observation = {  # dummy observation, the sim will end anyway
                "current_lidar": self.current_lidar,
                "prev_lidar1": self.prev_lidar1,
                "prev_lidar2": self.prev_lidar2,
                "prev_lidar3": self.prev_lidar3,
                "prev_lidar4": self.prev_lidar4,
                "prev_lidar5": self.prev_lidar5,
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
            self.prev_lidar1[:, 0] *= -1
            self.prev_lidar1 = self.current_lidar[::-1]
            self.prev_lidar2[:, 0] *= -1
            self.prev_lidar2 = self.current_lidar[::-1]
            self.prev_lidar3[:, 0] *= -1
            self.prev_lidar3 = self.current_lidar[::-1]
            self.prev_lidar4[:, 0] *= -1
            self.prev_lidar4 = self.current_lidar[::-1]
            self.prev_lidar5[:, 0] *= -1
            self.prev_lidar5 = self.current_lidar[::-1]

        observation = {
            "current_lidar": self.current_lidar,
            "prev_lidar1": self.prev_lidar1,
            "prev_lidar2": self.prev_lidar2,
            "prev_lidar3": self.prev_lidar3,
            "prev_lidar4": self.prev_lidar4,
            "prev_lidar5": self.prev_lidar5,
            "prev_throttle": self.prev_throttle,
            "prev_steering": self.prev_steering,
        }

        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        print("rendering")
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
        print("drawn")

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
            "MyVehicle",
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
            "MyVehicle",
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


def denormalize_action(action):
    """ Denormalize the throttle and steering from [0,1]² into [-1, 1]x[-0.5, 0.5] .


    Parameters
    ----------
    action : np.array
        A normalised action, where both throttle and steering are represented in [0,1]

    Returns
    -------
    denormalized_action : np.array
        An AirSim type action, where throttle is in [-1, 1] and steering in [-0.5, 0.5]

    """
    denormalized_action = (action - np.array([0.5, 0.5])) * np.array([2, 1])
    return denormalized_action


def normalize_action(action):
    """ Normalize throttle and steering from [-1, 1]x[-0.5, 0.5] into [0,1]².


    Parameters
    ----------
    action : TYPE
        An AirSim format action, where where throttle is in [-1, 1] and steering in [-0.5, 0.5]

    Returns
    -------
    normalize_action : nummpy array
        A normalised action, where both throttle and steering are represented in [0,1]

    """
    normalized_action = action * np.array([0.5, 1]) + np.array([0.5, 0.5])
    return normalized_action
