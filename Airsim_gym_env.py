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


class BoxAirSimEnv(gym.Env):
    """Custom AirSim Environment with Box action space that follows gym interface"""

    def __init__(self, client, lidar_size, dt, ClockSpeed,UE_spawn_point, liste_checkpoints_coordinates, liste_spawn_point,is_rendered=False):
        """
        

        Parameters
        ----------
        client : TYPE
            AirSim client.
        lidar_size : int
            Number of point observed by the lidar in each observation.
        dt : float
            Simulation time step separating two observations/actions.
        ClcokSpeed : int
            ClockSpeed selected in SETTINGS.JSON. Make sure that FPS/ClockSpeed > 30
        is_rendered : Boolean, optional
            Wether or not the lidar map is rendering. The default is False.

        Returns
        -------
        None.

        """
        super(BoxAirSimEnv, self).__init__()
        
		# Define action and observation space
            
        self.client=client
        self.car_controls = airsim.CarControls()
        self.car_state = client.getCarState() 
        # car_state is an AirSim object that contains informations that are not
        # obtainable in real experiments. Therefore, it cannot be used as a 
        # MDP object. However, it is useful for computing the reward
        
        
        self.is_rendered=is_rendered
        self.dt = dt
        self.ClockSpeed = ClockSpeed
        self.lidar_size=lidar_size
        
        self.total_reward=0
        self.done=False
        self.step_iterations=0
        
        self.UE_spawn_point = UE_spawn_point
        self.liste_checkpoints_coordonnes = liste_checkpoints_coordinates
        self.liste_spawn_point = liste_spawn_point
        
        
        
########## Below are MDP related objects #############
        
        self.action_space = gym.spaces.Box(low   = np.array([-1, -0.5], dtype=np.float32),
                                       high  = np.array([ 1,  0.5], dtype=np.float32),
                                       dtype=np.float32
                                       )
        
        # In this order
        # "throttle"
        # "steering"
        
		 #Example for using image as input (channel-first; channel-last also works):
             
        low =np.zeros((lidar_size,2), dtype=np.float32)
        high=np.zeros((lidar_size,2), dtype=np.float32)
        
        low[:,0]=-np.pi
        low[:,1]=0
        
        high[:,0]=np.pi
        high[:,1]=np.inf
        
        self.observation_space = gym.spaces.Dict(spaces={
            "current_lidar" : gym.spaces.Box(low=low, high=high, shape=(lidar_size,2), dtype=np.float32), # the format is [angle , radius]
            "prev_lidar"    : gym.spaces.Box(low=low, high=high, shape=(lidar_size,2), dtype=np.float32),
            
            "prev_throttle": gym.spaces.Box(low=-1  , high=1   , shape=(1,)),
            "prev_steering": gym.spaces.Box(low=-0.5, high=+0.5, shape=(1,))
            })
        
  
        
        
        
        
        
        
        
        
    def step(self, action):
        info={} #Just a debugging feature here
        
############ The actions are extracted from the "action" argument
        

        self.car_controls.throttle = float(action[0])
        self.car_controls.steering = float(action[1])

        self.client.setCarControls(self.car_controls)
        
        
    # Now that everything is good and proper, let's run  AirSim a bit
        self.client.simPause(False)
        time.sleep(self.dt/self.ClockSpeed) #TODO a dedicated thread may be more efficient
        self.client.simPause(True)
        

    # Get the state from AirSim
        self.car_state = self.client.getCarState()
        
        self.prev_throttle = np.array([action[0]])
        self.prev_steering = np.array([action[1]])
        
        position = self.car_state.kinematics_estimated.position
        gate_passed, finished_race = self.Circuit1.cycle_tick(position.x_val, position.y_val) #updating the checkpoint situation



    
############ extracts the observation ################
        self.current_lidar = convert_lidar_data_to_polar(self.client.getLidarData())

        # TODO : is copy padding really the best thing to do ? 
        ### Data padding if lidar is too short on this step
        current_lidar=self.current_lidar
        n_points_received = current_lidar.shape[0]
        if n_points_received < self.lidar_size : #not enough points !
            temp = np.ones((self.lidar_size-n_points_received+1 , 2))*current_lidar[0] #lets copy the first value multiple time
            adapted_lidar = np.concatenate((self.current_lidar,temp)) 
            self.current_lidar = adapted_lidar
            info["Reshaped lidar"]=True
        else:
            info["Reshaped lidar"]=False # lets inform that it happened for latter debugging
        #############################################
        
        
        observation = {
            "current_lidar" : self.current_lidar[:self.lidar_size,0:2], # if we have too many points the last ones are removed
            "prev_lidar"   : self.prev_lidar[:self.lidar_size,0:2],
            "prev_throttle" : self.prev_throttle,
            "prev_steering" : self.prev_steering
            }

        self.prev_lidar = self.current_lidar
        
############# Updates the reward ###############
        # collision info is necessary to compute reward
        collision_info = self.client.simGetCollisionInfo()
        crash = collision_info.has_collided
        
        reward = 0
        if crash:
            reward =-100
            
            
        elif self.car_state.speed <=1: # Lets force the car to move
            reward = -0.1

        if gate_passed : 
            reward += 50
            print("gate_passed")
        
        self.total_reward = self.total_reward + reward
        
    
        
        if crash:
            self.done=True
            print("Crash occured")
            print("Episode reward : " + str(self.total_reward) + 2*'\n')
        
    
    # displays ( or not ) the lidar observation
        if self.is_rendered:
            self.render()


        return observation, reward, self.done, info











    def reset(self):

        self.client.reset()


        # be careful, the call arguments for quaterninons are x,y,z,w 
        

        
        Circuit_wrapper1=Circuit_wrapper(self.liste_spawn_point, self.liste_checkpoints_coordonnes, UE_spawn_point=self.UE_spawn_point)
        spawn_point, theta, self.Circuit1 = Circuit_wrapper1.sample_random_spawn_point()
        
        
        
        x_val,y_val,z_val = spawn_point.x, spawn_point.y, spawn_point.z
        

        
        pose = airsim.Pose()
        
        orientation= airsim.Quaternionr (0,0,np.sin(theta/2)*1,np.cos(theta/2))
        position = airsim.Vector3r ( x_val, y_val, z_val)
        pose.position=position
        pose.orientation=orientation
        self.client.simSetVehiclePose(pose, ignore_collision=True)
        
    ##########

        self.throttle = 0
        self.steering = 0

        self.done = False
        
        self.total_reward=0
        self.client.simContinueForFrames( 100 ) #let's skip the first frames to inialise lidar and make sure everything is right
        time.sleep(1)  #the lidar data can take a bit of time before initialisation.
        
        self.current_lidar = convert_lidar_data_to_polar(self.client.getLidarData())
        self.prev_lidar = convert_lidar_data_to_polar(self.client.getLidarData())
        
        ### Data padding if lidar is too short on this step
        current_lidar=self.current_lidar
        n_points_received = current_lidar.shape[0]
        if n_points_received < self.lidar_size : #not enough points !
            temp = np.ones((self.lidar_size-n_points_received+1 , 2))*current_lidar[0] #lets copy the first value multiple time
            adapted_lidar = np.concatenate((self.current_lidar,temp)) 
            self.current_lidar = adapted_lidar
            self.prev_lidar = adapted_lidar
        ##############################################
        
        
        print("reset")

        self.prev_throttle = np.array([0])
        self.prev_steering = np.array([0])
        


        observation = {
            "current_lidar" : self.current_lidar[:self.lidar_size,0:2], # if we have too many points the last ones are removed
            "prev_lidar"    : self.prev_lidar[:self.lidar_size, 0:2],
            "prev_throttle" : self.prev_throttle,
            "prev_steering" : self.prev_steering
            }

        return observation  # reward, done, info can't be included







    def render(self, mode='human'):
        if not self.is_rendered:
            fig = plt.figure()
            self.ax = fig.add_subplot(projection='polar')
            self.is_rendered = True
            
        self.ax.clear()
        T=self.current_lidar[:,0]
        R=self.current_lidar[:,1]
        self.ax.scatter(T,R)
        plt.pause(0.01)
        plt.draw()
        
        ########### Image ###############
        responses = self.client.simGetImages([
        airsim.ImageRequest("Camera1", airsim.ImageType.Scene, False, False)], "MyVehicle")  #scene vision image in uncompressed RGB array
        response = responses[0]
    
        # get numpy array
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 
        
        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)
        
        cv2.imshow("image", img_rgb)
        
        

    def close (self):
         self.client.simPause(False)
         self.client.enableApiControl(False)
         


    def spawn_points_checker(self, wait_time=1):
        self.close()
        i=0
        for spawn_point in self.liste_spawn_point:
            print("Spawn : " + str(i)+'\n')
            theta_m = spawn_point.theta_min
            theta_M = spawn_point.theta_max            
            x_val,y_val,z_val = spawn_point.x, spawn_point.y, spawn_point.z
        

            pose = airsim.Pose()
            print('\tTheta min = '+str(theta_m))
            orientation= airsim.Quaternionr (0,0,np.sin(theta_m/2)*1,np.cos(theta_m/2))
            position = airsim.Vector3r ( x_val, y_val, z_val)
            pose.position=position
            pose.orientation=orientation
            self.client.simSetVehiclePose(pose, ignore_collision=True)
            time.sleep(wait_time)
            
            print('\tTheta max = '+str(theta_M))
            orientation= airsim.Quaternionr (0,0,np.sin(theta_M/2)*1,np.cos(theta_M/2))
            position = airsim.Vector3r ( x_val, y_val, z_val)
            pose.position=position
            pose.orientation=orientation
            self.client.simSetVehiclePose(pose, ignore_collision=True)
            time.sleep(wait_time)
            i+=1
            
            
            
class MultiDiscreteAirSimEnv(gym.Env):
    """Custom AirSim Environment MultiDiscrete action space that follows gym interface"""

    def __init__(self, client, lidar_size, dt, ClockSpeed,UE_spawn_point, liste_checkpoints_coordinates, liste_spawn_point,is_rendered=False):
        """
        

        Parameters
        ----------
        client : TYPE
            AirSim client.
        lidar_size : int
            Number of point observed by the lidar in each observation.
        dt : float
            Simulation time step separating two observations/actions.
        ClcokSpeed : int
            ClockSpeed selected in SETTINGS.JSON. Make sure that FPS/ClockSpeed > 30
        is_rendered : Boolean, optional
            Wether or not the lidar map is rendering. The default is False.

        Returns
        -------
        None.

        """
        super(MultiDiscreteAirSimEnv, self).__init__()
        
		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using discrete actions:
            
        self.client=client
        self.car_controls = airsim.CarControls()
        self.car_state = client.getCarState() 
        # car_state is an AirSim object that contains informations that are not
        # obtainable in real experiments. Therefore, it cannot be used as a 
        # MDP object. However, it is useful for computing the reward
        
        
        self.is_rendered=is_rendered
        self.dt = dt
        self.ClockSpeed = ClockSpeed
        self.lidar_size=lidar_size
        
        self.total_reward=0
        self.done=False
        self.step_iterations=0
        
        self.UE_spawn_point = UE_spawn_point
        self.liste_checkpoints_coordonnes = liste_checkpoints_coordinates
        self.liste_spawn_point = liste_spawn_point
        
        
        
########## Below are MDP related objects #############
        
        self.action_space = gym.spaces.MultiDiscrete([3,5])
        
        # In this order
        # "throttle" -> 0 = forward  ; 1 = stop        ; 2 = reverse
        # "steering" -> 0 = full left; 1 = middle left ; 2 = straight;
        #                            ; 3 = middle right; 4 = full right
        
        self.throttle_arg_to_throttle = {0:1 , 1:0 ,2:-1}
        self.steering_arg_to_steering = {0:-0.5 , 1:-0.25 , 2:0, 3:0.25, 4:0.5}
        
		 #Example for using image as input (channel-first; channel-last also works):
             
        low =np.zeros((lidar_size,2), dtype=np.float32)
        high=np.zeros((lidar_size,2), dtype=np.float32)
        
        low[:,0]=-np.pi
        low[:,1]=0
        
        high[:,0]=np.pi
        high[:,1]=np.inf
        
        self.observation_space = gym.spaces.Dict(spaces={
            "current_lidar" : gym.spaces.Box(low=low, high=high, shape=(lidar_size,2), dtype=np.float32), # the format is [angle , radius]
            "prev_lidar"    : gym.spaces.Box(low=low, high=high, shape=(lidar_size,2), dtype=np.float32),
            
            "prev_throttle": gym.spaces.Box(low=-1  , high=1   , shape=(1,)),
            "prev_steering": gym.spaces.Box(low=-0.5, high=+0.5, shape=(1,))
            })
        
  
        
        
        
        
        
        
        
        
    def step(self, action):
        info={} #Just a debugging feature here
        
############ The actions are extracted from the "action" argument
        throttle_arg = int(action[0])     
        steering_arg = int(action[1])
        

        self.car_controls.throttle = self.throttle_arg_to_throttle[throttle_arg]
        self.car_controls.steering = self.steering_arg_to_steering[steering_arg]

        self.client.setCarControls(self.car_controls)
        
        
    # Now that everything is good and proper, let's run  AirSim a bit
        self.client.simPause(False)
        time.sleep(self.dt/self.ClockSpeed) #TODO a dedicated thread may be more efficient
        self.client.simPause(True)
        

    # Get the state from AirSim
        self.car_state = self.client.getCarState()
        
        self.prev_throttle = np.array([self.throttle_arg_to_throttle[throttle_arg]])
        self.prev_steering = np.array([self.steering_arg_to_steering[steering_arg]])
        
        position = self.car_state.kinematics_estimated.position
        gate_passed, finished_race = self.Circuit1.cycle_tick(position.x_val, position.y_val) #updating the checkpoint situation



    
############ extracts the observation ################
        self.current_lidar = convert_lidar_data_to_polar(self.client.getLidarData())

        # TODO : is copy padding really the best thing to do ? 
        ### Data padding if lidar is too short on this step
        current_lidar=self.current_lidar
        n_points_received = current_lidar.shape[0]
        if n_points_received < self.lidar_size : #not enough points !
            temp = np.ones((self.lidar_size-n_points_received+1 , 2))*current_lidar[0] #lets copy the first value multiple time
            adapted_lidar = np.concatenate((self.current_lidar,temp)) 
            self.current_lidar = adapted_lidar
            info["Reshaped lidar"]=True
        else:
            info["Reshaped lidar"]=False # lets inform that it happened for latter debugging
        #############################################
        
        
        observation = {
            "current_lidar" : self.current_lidar[:self.lidar_size,0:2], # if we have too many points the last ones are removed
            "prev_lidar"   : self.prev_lidar[:self.lidar_size,0:2],
            "prev_throttle" : self.prev_throttle,
            "prev_steering" : self.prev_steering
            }

        self.prev_lidar = self.current_lidar
        
############# Updates the reward ###############
        # collision info is necessary to compute reward
        collision_info = self.client.simGetCollisionInfo()
        crash = collision_info.has_collided
        
        reward = 0
        if crash:
            reward =-100
            
            
        elif self.car_state.speed <=1: # Lets force the car to move
            reward = -0.1

        if gate_passed : 
            reward += 50
            print("gate_passed")
        
        self.total_reward = self.total_reward + reward
        
    
        
        if crash:
            self.done=True
            print("Crash occured")
            print("Episode reward : " + str(self.total_reward) + 2*'\n')
        
    
    # displays ( or not ) the lidar observation
        if self.is_rendered:
            self.render()


        return observation, reward, self.done, info











    def reset(self):

        self.client.reset()


        # be careful, the call arguments for quaterninons are x,y,z,w 
        

        
        Circuit_wrapper1=Circuit_wrapper(self.liste_spawn_point, self.liste_checkpoints_coordonnes, UE_spawn_point=self.UE_spawn_point)
        spawn_point, theta, self.Circuit1 = Circuit_wrapper1.sample_random_spawn_point()
        
        
        
        x_val,y_val,z_val = spawn_point.x, spawn_point.y, spawn_point.z
        

        
        pose = airsim.Pose()
        
        orientation= airsim.Quaternionr (0,0,np.sin(theta/2)*1,np.cos(theta/2))
        position = airsim.Vector3r ( x_val, y_val, z_val)
        pose.position=position
        pose.orientation=orientation
        self.client.simSetVehiclePose(pose, ignore_collision=True)
        
    ##########

        self.throttle = 0
        self.steering = 0

        self.done = False
        
        self.total_reward=0
        self.client.simContinueForFrames( 100 ) #let's skip the first frames to inialise lidar and make sure everything is right
        time.sleep(1)  #the lidar data can take a bit of time before initialisation.
        
        self.current_lidar = convert_lidar_data_to_polar(self.client.getLidarData())
        self.prev_lidar = convert_lidar_data_to_polar(self.client.getLidarData())
        
        ### Data padding if lidar is too short on this step
        current_lidar=self.current_lidar
        n_points_received = current_lidar.shape[0]
        if n_points_received < self.lidar_size : #not enough points !
            temp = np.ones((self.lidar_size-n_points_received+1 , 2))*current_lidar[0] #lets copy the first value multiple time
            adapted_lidar = np.concatenate((self.current_lidar,temp)) 
            self.current_lidar = adapted_lidar
            self.prev_lidar = adapted_lidar
        ##############################################
        
        
        print("reset")

        self.prev_throttle = np.array([0])
        self.prev_steering = np.array([0])
        


        observation = {
            "current_lidar" : self.current_lidar[:self.lidar_size,0:2], # if we have too many points the last ones are removed
            "prev_lidar"    : self.prev_lidar[:self.lidar_size, 0:2],
            "prev_throttle" : self.prev_throttle,
            "prev_steering" : self.prev_steering
            }

        return observation  # reward, done, info can't be included







    def render(self, mode='human'):
        if not self.is_rendered:
            fig = plt.figure()
            self.ax = fig.add_subplot(projection='polar')
            self.is_rendered = True
            
        self.ax.clear()
        T=self.current_lidar[:,0]
        R=self.current_lidar[:,1]
        self.ax.scatter(T,R)
        plt.pause(0.01)
        plt.draw()
        
        ########### Image ###############
        responses = self.client.simGetImages([
        airsim.ImageRequest("Camera1", airsim.ImageType.Scene, False, False)], "MyVehicle")  #scene vision image in uncompressed RGB array
        response = responses[0]
    
        # get numpy array
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 
        
        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)
        
        cv2.imshow("image", img_rgb)
        
        

    def close (self):
         self.client.simPause(False)
         self.client.enableApiControl(False)
         


    def spawn_points_checker(self, wait_time=1):
        self.close()
        i=0
        for spawn_point in self.liste_spawn_point:
            print("Spawn : " + str(i)+'\n')
            theta_m = spawn_point.theta_min
            theta_M = spawn_point.theta_max            
            x_val,y_val,z_val = spawn_point.x, spawn_point.y, spawn_point.z
        

            pose = airsim.Pose()
            print('\tTheta min = '+str(theta_m))
            orientation= airsim.Quaternionr (0,0,np.sin(theta_m/2)*1,np.cos(theta_m/2))
            position = airsim.Vector3r ( x_val, y_val, z_val)
            pose.position=position
            pose.orientation=orientation
            self.client.simSetVehiclePose(pose, ignore_collision=True)
            time.sleep(wait_time)
            
            print('\tTheta max = '+str(theta_M))
            orientation= airsim.Quaternionr (0,0,np.sin(theta_M/2)*1,np.cos(theta_M/2))
            position = airsim.Vector3r ( x_val, y_val, z_val)
            pose.position=position
            pose.orientation=orientation
            self.client.simSetVehiclePose(pose, ignore_collision=True)
            time.sleep(wait_time)
            i+=1
           
            
            






