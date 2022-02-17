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


import gym
from gym import spaces

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
    converted_lidar_data=np.array([theta1, ..., thetan]) , np.array([r1, ..., rn]).

    """
    list=lidar_data.point_cloud
    X=np.array(list[0::3])
    Y=np.array(list[1::3])
    
    R=np.sqrt(X**2+Y**2)
    T=np.arctan2(Y,X)

    # TODO
    # Could somebody add the 3rd dimension ?
    
    return T,R


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
            
        
        
        
########## Below are MDP related objects #############
        
        self.action_space = spaces.Dict(spaces={
            'throttle': spaces.Box(low=-1, high=1, shape=(1, 1)),
            'steering': spaces.Box(low=-0.5, high=0.5, shape=(1, 1)),
            'brake'   : spaces.Box(low=0, high=1, shape=(1, 1))
            })
        
		 #Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Dict(spaces={
            "current_lidar" : spaces.Box(low=0, high=np.inf, shape=(lidar_size, 2)),# the format is [angle , radius]
            "prev_lidar"   : spaces.Box(low=0, high=np.inf, shape=(lidar_size, 2)), #using the last observation could improve performances
            
            "prev_throttle": spaces.Box(low=-1, high=1, shape=(1, 1)),
            "prev_steering": spaces.Box(low=-0.5, high=+0.5, shape=(1,1)),
            "prev_brake"   : spaces.Box(low=0, high=1, shape=(1,1))
            })
        
  
        
        
        
        
        
        
        
        
    def step(self, action):
    
    


############ The actions are extracted from the "action" parameter

    # TODO Let's verify "action" has the correct format. There might already be a problem with negative steerings, as we would need to change gear manually
        self.car_controls.throttle = action['throttle']
        self.car_controls.steering = action['steering']
        self.car_controls.brake = action['brake']

        self.prev_action.append(action)


        client.setCarControls(self.car_controls)
        
        
    # Now that everything is good and proper, let's run AirSim a bit
        self.client.simContinueForFrames( self.dn )        
        
        
        
    # Waiting the simulation step to be over
        while not self.client.simIsPause():
            pass
        

    # Get the state from AirSim
        self.car_state = client.getCarState()



    # TODO
############ extracts the observation ################
        T, R = convert_lidar_data_to_polar(self.client.getLidarData())

        self.current_lidar = [T, R]
        self.prev_lidar = self.current_lidar

        observation = [self.current_lidar[0],
                       self.current_lidar[1]] + self.prev_action

    # TODO
############# Updates the reward ###############

        self.total_reward = self.total_reward + reward



    # TODO
    # Tests the break condition in case of crutial mistake, and applies a heavy
    # penalty
    
        if break_condition:
            done = True

        
    
    # displays ( or not ) the lidar observation
        if self.is_rendered:
            self.render()

    # TODO
    # What cool tricks could we do with this one ?
        info = {}

        return observation, reward, done, info











    def reset(self):

        client.reset()

# TODO
# to avoid overfitting, there must be a random positioning of the vehicule
# in the future

        self.throttle = 0
        self.steering = 0
        self.brake = 0

        self.done = False

        # depending on the client reset time, the lidar may not be
        time.sleep(1)
        # available. let's sleep a bit.
        T, R = convert_lidar_data_to_polar(self.client.getLidarData())

        self.current_lidar = [T, R]

        self.prev_throttle = 0
        self.prev_steering = 0
        self.prev_brake = 0

        self.prev_action = [self.prev_throttle,
                            self.prev_steering, self.prev_brake]

        observation = [self.current_lidar[0],
                       self.current_lidar[1]] + self.prev_action
        
        self.client.simContinueForFrames( self.dn ) # Starts the simulation

        return observation  # reward, done, info can't be included







    def render(self, mode='human'):
        if not self.is_rendered:
            fig = plt.figure()
            self.ax = fig.add_subplot(projection='polar')
            self.is_rendered = True
            
        self.ax.clear()
        [T,R]=self.current_lidar
        self.ax.scatter(T,R)
        plt.pause(0.01)
        plt.draw()
        
        
 
        

    def close (self):
         client.enableApiControl(False)














###############################################################################


# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.simPause(True)
client.enableApiControl(True)



airsim_env=CustomEnv(client, dn=50 ,lidar_size=10)
airsim_env.reset()
client.enableApiControl(False)


airsim_env.close()
