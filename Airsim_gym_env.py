# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:05:35 2022

@author: jamyl
"""

import numpy as np
import airsim
from stable_baselines3.common.env_checker import check_env
import time
import matplotlib.pyplot as plt
import cv2
import random


import gym
from gym import spaces

from stable_baselines3 import SAC

def convert_lidar_data_to_polar(lidar_data):
    """
    
    Parameters
    ----------
    lidar_data : TYPE LidarData
    
        Transforms the lidar data to convert it to a real life format, that is from
        (x_hit, y_hit, z_hit) to (angle_hit, distance_hit). Make sure to set
        "DatFrame": "SensorLocalFrame" in the settings.JSON to get relative
        coordinates from hit-points.
        
        Note : so far, only 2 dimensions lidar is supported. Thus, the Z coordinate
        will simply be ignored

    Returns
    -------
    converted_lidar_data=np.array([theta_1, ..., theta_n]) , np.array([r_1, ..., r_n]).

    """
    list=lidar_data.point_cloud
    X=np.array(list[0::3])
    Y=np.array(list[1::3])
    
    R=np.sqrt(X**2+Y**2)
    T=np.arctan2(Y,X)
    
    # TODO
    # Could somebody add the 3rd dimension ?
    
    return np.column_stack((T,R))


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, client, lidar_size, dn,is_rendered=False):
        """
        

        Parameters
        ----------
        client : TYPE
            AirSim client.
        lidar_size : int
            Number of point observed by the lidar in each observation.
        dn : float
            Number of frames during which each simulation step runs.
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
        self.dn = dn
        self.lidar_size=lidar_size
        
        self.total_reward=0
        self.done=False
        self.step_iterations=0
        
        
        
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
        
        
    # Now that everything is good and proper, let's run a few frames of AirSim
        self.client.simContinueForFrames( self.dn )        
        
        
    # Waiting the simulation step to be over
        while not self.client.simIsPause():
            pass

        

    # Get the state from AirSim
        self.car_state = client.getCarState()
        
        self.prev_throttle = np.array([action[0]])
        self.prev_steering = np.array([action[1]])



    
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
        
        
        if crash:
            reward =-100
            
            
        elif self.car_state.speed <=1: # Lets force the car to move
            reward = -0.1
            
        else:
            reward = 0.1*self.car_state.speed #the faster the better


        
        self.total_reward = self.total_reward + reward



    # TODO
    # Tests the break condition in case of crucial mistake, and applies a heavy
    # penalty or not
        
    
        break_condition=False #placeholder
        position = self.car_state.kinematics_estimated.position
        if position.x_val**2+position.y_val**2 >= 20000:
            print('breaking out')
            break_condition=True
        

        
        if break_condition:
            self.done = True
            print("Episode reward : " + str(self.total_reward) + 2*'\n')
        
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


        x_val=random.uniform(-30, 30)
        y_val=random.uniform(-30, 30)
        
        theta=random.uniform(0, 2*np.pi)
        
        pose = airsim.Pose()
        
        orientation= airsim.Quaternionr (0,0,np.sin(theta/2)*1,np.cos(theta/2))
        position = airsim.Vector3r ( x_val, y_val,-1)
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
         client.enableApiControl(False)








###############################################################################

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.simPause(True)
client.enableApiControl(True)

airsim_env=CustomEnv(client, dn=10 ,lidar_size=500)
# Testing the model after learning

model = SAC.load("C:/Users/jamyl/Documents/GitHub/AirSim-GYM-environment/Training/32800",tensorboard_log="C:/Users/jamyl/Documents/GitHub/AirSim-GYM-environment/Training")
obs = airsim_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = airsim_env.step(action)
    airsim_env.render()
    if done:
      obs = airsim_env.reset()

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





#%% Trainign a model
models_dir = "C:/Users/jamyl/Desktop/TER_dossier/Training"
logdir = "C:/Users/jamyl/Desktop/TER_dossier/Training"

TIMESTEPS=100
model = SAC("MultiInputPolicy", airsim_env, verbose=1,tensorboard_log=logdir)
iters=0
while(True):
    iters=iters+1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="SAC")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
    
    



#%% checking SB3 compatibility

check_env(airsim_env)