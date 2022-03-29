# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:05:35 2022

@author: jamyl
"""

import numpy as np
import airsim
from tqdm import tqdm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.save_util import save_to_pkl
from stable_baselines3.common import utils
import time
import matplotlib.pyplot as plt
import cv2
import random

import sys
sys.path.append("C:/Users/jamyl/Documents/GitHub/AirSim-GYM-environment")
from jamys_toolkit import Circuit_wrapper, convert_lidar_data_to_polar, Circuit_spawn, fetch_action, pre_train



import gym
from gym import spaces

from stable_baselines3 import SAC
SAC.pre_train = pre_train # Adding my personal touch ;)


import sys
sys.path.append("C:/Users/jamyl/Documents/GitHub/AirSim-GYM-environment/jamys_toolkit.py")
    


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

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
        super(CustomEnv, self).__init__()
        
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
        self.liste_checkpoints_coordonnes = liste_checkpoints_coordonnes
        self.liste_spawn_point = liste_spawn_point
        
        
        
########## Below are MDP related objects #############
        
        self.action_space = spaces.Box(low   = np.array([-1, -0.5], dtype=np.float32),
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
        
        self.observation_space = spaces.Dict(spaces={
            "current_lidar" : spaces.Box(low=low, high=high, shape=(lidar_size,2), dtype=np.float32), # the format is [angle , radius]
            "prev_lidar"    : spaces.Box(low=low, high=high, shape=(lidar_size,2), dtype=np.float32),
            
            "prev_throttle": spaces.Box(low=-1  , high=1   , shape=(1,)),
            "prev_steering": spaces.Box(low=-0.5, high=+0.5, shape=(1,))
            })
        
  
        
        
        
        
        
        
        
        
    def step(self, action):
        info={} #Just a debugging feature here
        
############ The actions are extracted from the "action" argument

    #TODO There might already be a problem with negative throttle, as we would need to change gear manually
        

        self.car_controls.throttle = float(action[0])
        self.car_controls.steering = float(action[1])

        A = self.car_controls
        self.client.setCarControls(A)
        
        
    # Now that everything is good and proper, let's run  AirSim a bit
        self.client.simPause(False)
        time.sleep(self.dt/self.ClockSpeed) #TODO a dedicated thread may be more efficient
        self.client.simPause(True)
        

    # Get the state from AirSim
        self.car_state = client.getCarState()
        
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
    # TODO
############# Updates the reward ###############
        # collision info is necessary to compute reward
        collision_info = client.simGetCollisionInfo()
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



    # TODO
    # Tests the break condition in case of crucial mistake, and applies a heavy
    # penalty or not
        
    
        
        if crash:
            self.done=True
            print("Crash occured")
            print("Episode reward : " + str(self.total_reward) + 2*'\n')
        
    
    # displays ( or not ) the lidar observation
        if self.is_rendered:
            self.render()


        return observation, reward, self.done, info











    def reset(self):

        client.reset()

# TODO
# to avoid overfitting, there must be a random positioning of the vehicule
# in the future


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
        responses = client.simGetImages([
        airsim.ImageRequest("Camera1", airsim.ImageType.Scene, False, False)], "MyVehicle")  #scene vision image in uncompressed RGB array
        response = responses[0]
    
        # get numpy array
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 
        
        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)
        
        cv2.imshow("image", img_rgb)
        
        

    def close (self):
         client.simPause(False)
         client.enableApiControl(False)
         


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
            
            
            






###############################################################################
#RC circuit model lidar only branch

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.simPause(True)
client.enableApiControl(True)

spawn = np.array([-13316.896484, 4559.699707, 322.200134])


liste_checkpoints_coordonnes=[[-10405.0, 4570.0, 10],
                              [-8845.0, 3100.0, 10],
                              [-9785.0, 1420.0, 10],
                              [-9795.0, -320.0, 10],
                              [-7885.0, -2060.0, 10],
                              [-11525.0, -2060.0, 10],
                              [-12275.0, -3800.0, 10.91],
                              [-12755.0, -6880.0, 10],
                              [-15405.0, -5640.0, 10],
                              [-17145.0, -3900.0, 10],
                              [-18965.0, -3550.0, 10],
                              [-18915.0, -270.0, 10],
                              [-17655.0, 1540.0, 10],
                              [-18915.0, 3560.0, 10],
                              [-17355.0, 4580.0, 10],
                              [-14255.0, 4580.0, 10]
                              ]



def create_spawn_points(spawn): #Just a way to hide this big part
    liste_spawn_point=[]
    ##################### 0 #################
    spawn1= Circuit_spawn(-13650, 4920, 350, -np.pi/4, np.pi/4,checkpoint_index=0,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-13030, 4240, 350, -np.pi/4, np.pi/4,checkpoint_index=0,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    spawn3= Circuit_spawn(-12230, 4710, 350, -np.pi/4, np.pi/4,checkpoint_index=0,spawn_point=spawn)
    liste_spawn_point.append(spawn3)
    
    spawn4= Circuit_spawn(-11800, 4210, 350, -np.pi/8, np.pi/4,checkpoint_index=0,spawn_point=spawn)
    liste_spawn_point.append(spawn4)
    
    spawn5= Circuit_spawn(-11220, 4890, 350, -np.pi/4, np.pi/4,checkpoint_index=0,spawn_point=spawn)
    liste_spawn_point.append(spawn5)
    
    ######### 1 ################
    spawn1= Circuit_spawn(-9880, 4890, 350, -np.pi/4, np.pi/4,checkpoint_index=1,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-9720, 4280, 350, -np.pi/4, np.pi/4,checkpoint_index=1,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    spawn3= Circuit_spawn(-9470, 4580, 350, -np.pi/3, np.pi/4,checkpoint_index=1,spawn_point=spawn)
    liste_spawn_point.append(spawn3)
    
    spawn4= Circuit_spawn(-9130, 3720, 350, -np.pi/2-np.pi/4, -np.pi/2+np.pi/4,checkpoint_index=1,spawn_point=spawn)
    liste_spawn_point.append(spawn4)
    
    spawn5= Circuit_spawn(-8740, 3720, 350, -np.pi/2-np.pi/4, -np.pi/2+np.pi/4,checkpoint_index=1,spawn_point=spawn)
    liste_spawn_point.append(spawn5)
    
    ######### 2 ################
    spawn1= Circuit_spawn(-9130, 2470, 350, -np.pi/2-np.pi/4, -np.pi/2+np.pi/4,checkpoint_index=2,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-8550, 2470, 350, -np.pi/2-np.pi/4, -np.pi/2+np.pi/4,checkpoint_index=2,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    spawn3= Circuit_spawn(-8550, 1650, 350, -np.pi/2-np.pi/3, -np.pi/2,checkpoint_index=2,spawn_point=spawn)
    liste_spawn_point.append(spawn3)
    
    ######### 3 ################
    spawn1= Circuit_spawn(-10430, 1650, 350, -np.pi, -np.pi+np.pi/2,checkpoint_index=3,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-10380, 930, 350, -np.pi+np.pi/6, -np.pi+np.pi/2,checkpoint_index=3,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    spawn3= Circuit_spawn(-11080, 910, 350, -np.pi/2-np.pi/4, -np.pi/2+np.pi/4,checkpoint_index=3,spawn_point=spawn)
    liste_spawn_point.append(spawn3)
    
    spawn4= Circuit_spawn(-10540, 130, 350, -np.pi/2+np.pi/6, -np.pi/2+np.pi/2,checkpoint_index=3,spawn_point=spawn)
    liste_spawn_point.append(spawn4)
    
    spawn5= Circuit_spawn(-11120, -500, 350, -np.pi/6, np.pi/6,checkpoint_index=3,spawn_point=spawn)
    liste_spawn_point.append(spawn5)
    
    spawn6= Circuit_spawn(-10390, -630, 350, -np.pi/6, np.pi/4,checkpoint_index=3,spawn_point=spawn)
    liste_spawn_point.append(spawn6)
    
    ######### 4 ################
    spawn1= Circuit_spawn(-9170, -80, 350, -np.pi/4, np.pi/4,checkpoint_index=4,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-8590, -560, 350, -np.pi/4, np.pi/4,checkpoint_index=4,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    spawn3= Circuit_spawn(-8020, 10, 350, -np.pi/4, np.pi/6,checkpoint_index=4,spawn_point=spawn)
    liste_spawn_point.append(spawn3)
    
    spawn4= Circuit_spawn(-7790, -640, 350, -np.pi/6, np.pi/4,checkpoint_index=4,spawn_point=spawn)
    liste_spawn_point.append(spawn4)
    
    spawn5= Circuit_spawn(-7100, -840, 350, -np.pi/2, -np.pi/2+np.pi/4,checkpoint_index=4,spawn_point=spawn)
    liste_spawn_point.append(spawn5)
    
    spawn6= Circuit_spawn(-6430, -1450, 350, -np.pi/2-np.pi/3, -np.pi/2,checkpoint_index=4,spawn_point=spawn)
    liste_spawn_point.append(spawn6)
    
    spawn7= Circuit_spawn(-7170, -1680, 350, -np.pi/2-np.pi/3, -np.pi/2-np.pi/6,checkpoint_index=4,spawn_point=spawn)
    liste_spawn_point.append(spawn7)
    
    spawn8= Circuit_spawn(-6820, -2350, 350, -np.pi-np.pi/4, -np.pi,checkpoint_index=4,spawn_point=spawn)
    liste_spawn_point.append(spawn8)
    
    ######### 5 ################
    spawn1= Circuit_spawn(-8540, -1800, 350, -np.pi-np.pi/4, -np.pi+np.pi/4,checkpoint_index=5,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-8580, -2440, 350, -np.pi-np.pi/4, -np.pi+np.pi/4,checkpoint_index=5,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    spawn3= Circuit_spawn(-9150, -1960, 350, -np.pi-np.pi/4, -np.pi+np.pi/4,checkpoint_index=5,spawn_point=spawn)
    liste_spawn_point.append(spawn3)
    
    spawn4= Circuit_spawn(-9780, -2410, 350, -np.pi-np.pi/4, -np.pi+np.pi/4,checkpoint_index=5,spawn_point=spawn)
    liste_spawn_point.append(spawn4)
    
    spawn5= Circuit_spawn(-10290, -1800, 350, -np.pi-np.pi/4, -np.pi+np.pi/4,checkpoint_index=5,spawn_point=spawn)
    liste_spawn_point.append(spawn5)
    
    spawn6= Circuit_spawn(-10880, -2340, 350, -np.pi-np.pi/4, -np.pi+np.pi/4,checkpoint_index=5,spawn_point=spawn)
    liste_spawn_point.append(spawn6)
    
    ######### 6 ################
    spawn1= Circuit_spawn(-12200, -2500, 350, -np.pi+np.pi/4, -np.pi+np.pi/3,checkpoint_index=6,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-12540, -1860, 350, -np.pi+np.pi/4, -np.pi+np.pi/2,checkpoint_index=6,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    spawn3= Circuit_spawn(-13020, -3150, 350, -np.pi+np.pi/2, -np.pi-np.pi/3,checkpoint_index=6,spawn_point=spawn)
    liste_spawn_point.append(spawn3)
    
    ######### 7 ################
    spawn1= Circuit_spawn(-11410, -3780, 350, -np.pi/2-np.pi/4, -np.pi/2+np.pi/6,checkpoint_index=7,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-11790, -4880, 350, -np.pi/2-np.pi/6, -np.pi/2+np.pi/4,checkpoint_index=7,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    spawn3= Circuit_spawn(-11240, -5410, 350, -np.pi/2-np.pi/4, -np.pi/2,checkpoint_index=7,spawn_point=spawn)
    liste_spawn_point.append(spawn3)
    
    spawn4= Circuit_spawn(-11920, -5950, 350, -np.pi/2, -np.pi/2+np.pi/6,checkpoint_index=7,spawn_point=spawn)
    liste_spawn_point.append(spawn4)
    
    spawn5= Circuit_spawn(-11210, -6270, 350, -np.pi/2-np.pi/4, -np.pi/2,checkpoint_index=7,spawn_point=spawn)
    liste_spawn_point.append(spawn5)
    
    spawn6= Circuit_spawn(-11680, -6750, 350, -np.pi-np.pi/6, -np.pi+np.pi/4,checkpoint_index=7,spawn_point=spawn)
    liste_spawn_point.append(spawn6)
    
    ######### 8 ################
    spawn1= Circuit_spawn(-13450, -7210, 350, -np.pi-np.pi/4, -np.pi+np.pi/6,checkpoint_index=8,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-13890, -6680, 350, -np.pi-np.pi/6, -np.pi+np.pi/4,checkpoint_index=8,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    spawn3= Circuit_spawn(-14650, -7100, 350, -np.pi-np.pi/4, -np.pi+np.pi/4,checkpoint_index=8,spawn_point=spawn)
    liste_spawn_point.append(spawn3)
    
    spawn4= Circuit_spawn(-15070, -6640, 350, -3*np.pi/2, -3*np.pi/2+np.pi/4,checkpoint_index=8,spawn_point=spawn)
    liste_spawn_point.append(spawn4)
    
    ######### 9 ################
    spawn1= Circuit_spawn(-15680, -5030, 350, -3*np.pi/2-np.pi/4, -3*np.pi/2+np.pi/6,checkpoint_index=9,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-15150, -4810, 350, -3*np.pi/2-np.pi/4, -3*np.pi/2+np.pi/4,checkpoint_index=9,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    spawn3= Circuit_spawn(-15500, -4210, 350, -3*np.pi/2+np.pi/6, -3*np.pi/2+np.pi/4,checkpoint_index=9,spawn_point=spawn)
    liste_spawn_point.append(spawn3)
    
    spawn4= Circuit_spawn(-16020, -3570, 350, -np.pi-np.pi/4, -np.pi+np.pi/6,checkpoint_index=9,spawn_point=spawn)
    liste_spawn_point.append(spawn4)
    
    spawn5= Circuit_spawn(-16800, -3140, 350,  -np.pi, -np.pi+np.pi/2,checkpoint_index=9,spawn_point=spawn)
    liste_spawn_point.append(spawn5)
    
    ######### 10 ################
    spawn1= Circuit_spawn(-16940, -4550, 350, -np.pi, -np.pi/2,checkpoint_index=10,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-17150, -5150, 350, -np.pi, -np.pi-np.pi/4,checkpoint_index=10,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    spawn3= Circuit_spawn(-17640, -4790, 350, -np.pi+np.pi/4, -np.pi-np.pi/4,checkpoint_index=10,spawn_point=spawn)
    liste_spawn_point.append(spawn3)
    
    spawn4= Circuit_spawn(-18450, -4880, 350, -np.pi-np.pi/2, -np.pi-np.pi/4,checkpoint_index=10,spawn_point=spawn)
    liste_spawn_point.append(spawn4)
    
    spawn5= Circuit_spawn(-19050, -4420, 350, -3*np.pi/2-np.pi/4, -3*np.pi/2+np.pi/4,checkpoint_index=10,spawn_point=spawn)
    liste_spawn_point.append(spawn5)
    
    
    ######### 11 ################
    spawn1= Circuit_spawn(-19260, -2900, 350, -3*np.pi/2-np.pi/4, -3*np.pi/2,checkpoint_index=11,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-18690, -2750, 350, -3*np.pi/2-np.pi/4, -3*np.pi/2 + np.pi/4,checkpoint_index=11,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    spawn3= Circuit_spawn(-19090, -2170, 350, -3*np.pi/2-np.pi/4, -3*np.pi/2+ np.pi/4,checkpoint_index=11,spawn_point=spawn)
    liste_spawn_point.append(spawn3)
    
    spawn4= Circuit_spawn(-18700, -1670, 350, -3*np.pi/2-np.pi/4, -3*np.pi/2+ np.pi/4,checkpoint_index=11,spawn_point=spawn)
    liste_spawn_point.append(spawn4)
    
    spawn5= Circuit_spawn(-19200, -1240, 350, -3*np.pi/2-np.pi/4, -3*np.pi/2+ np.pi/4,checkpoint_index=11,spawn_point=spawn)
    liste_spawn_point.append(spawn5)
    
    ######### 12 ################
    spawn1= Circuit_spawn(-19090, 570, 350, -np.pi/6, np.pi/6,checkpoint_index=12,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-18250, 480, 350, 0, np.pi/3,checkpoint_index=12,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    ######### 13 ################
    spawn1= Circuit_spawn(-18060, 2230, 350, -np.pi-np.pi/4, -np.pi+np.pi/6,checkpoint_index=13,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-19040, 2140, 350, -3*np.pi/4-np.pi/4, -3*np.pi/2,checkpoint_index=13,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    ######### 14 ################
    spawn1= Circuit_spawn(-19210, 4380, 350, -np.pi/6, np.pi/4,checkpoint_index=14,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-18650, 4820, 350, -np.pi/6, np.pi/4,checkpoint_index=14,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    spawn3= Circuit_spawn(-18200, 4360, 350, -np.pi/6, np.pi/4,checkpoint_index=14,spawn_point=spawn)
    liste_spawn_point.append(spawn3)
    
    ######### 15 ################
    spawn1= Circuit_spawn(-16600, 4890, 350, -np.pi/4, np.pi/6,checkpoint_index=14,spawn_point=spawn)
    liste_spawn_point.append(spawn1)
    
    spawn2= Circuit_spawn(-16210, 4270, 350, -np.pi/6, np.pi/4,checkpoint_index=14,spawn_point=spawn)
    liste_spawn_point.append(spawn2)
    
    spawn3= Circuit_spawn(-15780, 4860, 350, -np.pi/4, np.pi/6,checkpoint_index=14,spawn_point=spawn)
    liste_spawn_point.append(spawn3)
    
    spawn4= Circuit_spawn(-15100, 4270, 350, -np.pi/6, np.pi/4,checkpoint_index=14,spawn_point=spawn)
    liste_spawn_point.append(spawn4)
    return liste_spawn_point
liste_spawn_point = create_spawn_points(spawn)

ClockSpeed = 1
airsim_env=CustomEnv(client, dt=0.1, ClockSpeed=ClockSpeed ,lidar_size=200,
                      UE_spawn_point=spawn,
                      liste_checkpoints_coordinates = liste_checkpoints_coordonnes,
                      liste_spawn_point = liste_spawn_point)



obs = airsim_env.reset()
while(True):
    steer_coeff = 1/20
    speed_coeff = 1

    while True:
        dots = np.array(obs['current_lidar'])
        left_dists = dots[:,:][50 > dots[:,0]]
        left_dists = left_dists[:,1][left_dists[:,1] > 40]
        left_dist = np.mean(left_dists)
        right_dists = dots[:,:][dots[:,0] > 310 ]
        right_dists = right_dists[:,1][right_dists[:,0] < 320 ]
        right_dist = np.mean(right_dists)
        front_dists = np.concatenate((dots[:,1][dots[:,0] < 5],dots[:,1][dots[:,0] > 355]))
        front_dist = np.mean(front_dists)
        
        
        if front_dists.size == 0:
            motor_speed = 0
        else:
            motor_speed = speed_coeff*front_dist
            
        if right_dists.size == 0:
            steering = 0.5
        elif left_dists.size == 0:
            steering = -0.5
        else:
            steering = steer_coeff*(left_dist - right_dist)
            
        throttle = motor_speed/10
        action = np.array([throttle, -steering])
        obs, reward, done, info = airsim_env.step(action)
    
        if done :
            airsim_env.reset()
        
        
        
        

    


#%%#########
def find_radius(angle, lidar):
    boolean1 = lidar[:,0] >= angle
    boolean2 = np.roll(boolean1,1)
    boolean3 = np.logical_xor(boolean1, boolean2)
    try :
        index = np.where(boolean3[1:] == True)[0][0]
        return lidar[index+1, 1]
    except :
        return None



obs = airsim_env.reset()




prev_steering = 0
steering = 0
while True:
    right_distance = find_radius(np.pi/4, obs['current_lidar']) 
    left_distance  = find_radius(-np.pi/4, obs['current_lidar']) 
    throttle = 0.5
    if right_distance != None and left_distance != None :
        prev_steering = steering
        steering = (right_distance - left_distance)/20
    else : 
        steering = prev_steering
    
    if steering >=0 :
        steering = 0.25
    else:
        steering =-0.25
    
    action = np.array([throttle, steering])
    obs, reward, done, info = airsim_env.step(action)
    
    if done :
        airsim_env.reset()





#%% Old circuit

ClockSpeed = 3

airsim_env=CustomEnv(client, dt=0.1, ClockSpeed=ClockSpeed ,lidar_size=200,
                      UE_spawn_point=spawn,
                      liste_checkpoints_coordinates = liste_checkpoints_coordonnes,
                      liste_spawn_point = liste_spawn_point)

path = "C:/Users/jamyl/Desktop/DUMP/'filename_pi.obj'"




#%% basic control

def find_radius(angle, lidar):
    boolean1 = lidar[:,0] >= angle
    boolean2 = np.roll(boolean1,1)
    boolean3 = np.logical_xor(boolean1, boolean2)
    try :
        index = np.where(boolean3[1:] == True)[0][0]
        return lidar[index+1, 1]
    except :
        return None



obs = airsim_env.reset()




prev_steering = 0
steering = 0
while True:
    right_distance = find_radius(np.pi/2, obs['current_lidar']) 
    left_distance  = find_radius(-np.pi/2, obs['current_lidar']) 
    throttle = 0.5
    if right_distance != None and left_distance != None :
        prev_steering = steering
        steering = (right_distance - left_distance)/20
    else : 
        steering = prev_steering
    
    if steering >=0 :
        steering = 0.25
    else:
        steering =-0.25
    
    action = np.array([throttle, steering])
    obs, reward, done, info = airsim_env.step(action)
    
    if done :
        airsim_env.reset()



#%% Trainign a model
models_dir = "P:/Training/Training_V4"
logdir = "P:/Training/Training_V4"

from jamys_toolkit import Jamys_CustomFeaturesExtractor

policy_kwargs = dict(
    features_extractor_class=Jamys_CustomFeaturesExtractor,
    features_extractor_kwargs=dict(Lidar_data_label=["current_lidar", "prev_lidar"],
                                    lidar_output_dim=100)
)

TIMESTEPS=1000

model = SAC("MultiInputPolicy", airsim_env,
            verbose=1,tensorboard_log=logdir,
            policy_kwargs=policy_kwargs)

#model.pre_train(replay_buffer_path = path)

iters=0
while(True):
    iters=iters+1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="SAC_Lidar_only_RC")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
    

#%% Testing the model after learning

model = SAC.load("P:/Training_V1/504400",tensorboard_log="P:/Training_V1")
obs = airsim_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = airsim_env.step(action)
    if done:
      obs = airsim_env.reset()



    
#%% pretraining the model on prerecorded master trajectories

models_dir = "C:/Users/jamyl/Desktop/TER_dossier/Training"
logdir = "C:/Users/jamyl/Desktop/TER_dossier/Training"



TIMESTEPS=100
model = SAC("MultiInputPolicy", airsim_env, verbose=1,tensorboard_log=logdir)

model.pre_train(replay_buffer_path = path)







#%% Loading the replay buffer and training on that
models_dir = "C:/Users/jamyl/Desktop/TER_dossier/Training"
logdir = "C:/Users/jamyl/Desktop/TER_dossier/Training"

TIMESTEPS=100
model = SAC("MultiInputPolicy", airsim_env, verbose=1,tensorboard_log=logdir) 
model.load_replay_buffer(path)
model.replay_buffer.device='cuda' # TODO is it really working ??
buffer = model.replay_buffer

iters=0
while(True):
    iters=iters+1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="SAC_Lidar_only_RC")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
    



#%% gathering teacher data to save in a replay buffer

playing_time = 300 #in seconds



airsim_env.reset()
airsim_env.close()



replay_buffer = DictReplayBuffer(
                    buffer_size = 1_000_000,
                    observation_space = airsim_env.observation_space,
                    action_space = airsim_env.action_space
                )


client.simPause(True)
action = fetch_action(client)
observation, reward, done, info = airsim_env.step(np.array([0,0]))
starting_time = time.time()
while(True): 
    action = fetch_action(client)
    future_observation, future_reward, future_done, info = airsim_env.step(np.array([0,0]))
    replay_buffer.add(observation, next_obs = future_observation, action = action,
                      reward = reward, done = done, infos = [info])
    observation, reward, done = future_observation, future_reward, future_done
    if done :
        airsim_env.reset()
        
    if time.time()-starting_time >=playing_time:
        break

print("saving...")
save_to_pkl(path, replay_buffer)





#%% Playing on the circuit
airsim_env.reset()
airsim_env.close()

while(True):   
    airsim_env.client.car_state = airsim_env.client.getCarState()
    position = airsim_env.client.car_state.kinematics_estimated.position
    gate, finish = airsim_env.Circuit1.cycle_tick(position.x_val, position.y_val)
    collision_info = client.simGetCollisionInfo()
    crash = collision_info.has_collided
    if crash:
        print("crash")
        airsim_env.reset()
        airsim_env.close()
        
    if gate :
        print("gate_passed")
    if finish :
        print("finish")
        break



    


#%% Load a previously trained model
model = SAC.load("C:/Users/jamyl/Documents/GitHub/AirSim-GYM-environment/Training/32800",tensorboard_log="C:/Users/jamyl/Documents/GitHub/AirSim-GYM-environment/Training")


model.set_env(airsim_env)
airsim_env.reset()
airsim_env.render()

while(True):
    model.learn(total_timesteps=1000, reset_num_timesteps=False, tb_log_name="SAC_1")


#%% Making random control

airsim_env.reset()
airsim_env.render()

for _ in range(10000):



    
    
    action=np.random.rand(3)
    action[1]=(action[1]-0.5) #normalisation le steering doit être entre -0.5 et 0.5
    action[2]=0 #pas de freinage
    
    if _<10:
        action[1]=0.5
    airsim_env.step(action)








airsim_env.reset()
airsim_env.close()









#%% checking SB3 compatibility

check_env(airsim_env)