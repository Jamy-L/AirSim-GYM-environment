# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 08:50:14 2022

@author: jamyl
"""

import numpy as np
import random

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


def convert_global_to_relative_position(Ue_spawn_point, Ue_global_point):
    """
    Converts the coordinates given in UE to coordinates given by the airsim API
    Basically, it is just a 100 division and a inversion of the z axis
    (who is still not using direct coordinate system ???)

    Parameters
    ----------
    spawn_point : NUMPY ARRAY
        Coordinates of the spawn point given but UE (refer to UE 4 for that).
    global_point : NUMPY ARRAY
        Global coordinates of the point to convert (in UE).

    Returns
    -------
    x : float
    y : float
    z : float
        The position of the given point with regard to the spawn point.
        A discount by a 100 factor is done because the AirSim API uses a 
        different basis than UE4...

    """
    
    C = Ue_global_point - Ue_spawn_point
    C=C/100
    C[2]*=-1
    return C[0], C[1], C[2]

class Checkpoint():
    def __init__(self,x_pos, y_pos, radius, next_checkpoint = None, index=None):
        """
        

        Parameters
        ----------
        x_pos : TYPE float
            x coordinate in the airsim relative axis. Make sure to call
            convert_global_to_relative_position before if you have UE coordinates
        y_pos : TYPE float
            y coordinate.
        radius : TYPE float
            radius of the checkpoint.
        next_checkpoint : TYPE Checkpoint
            Next following checkpoint : where to go once this one is, passed ?
        index (optional) : for debugging purposes. Just a label

        Returns
        -------
        None.

        """
        self.x = x_pos
        self.y = y_pos
        self.r = radius
        self.next_checkpoint = next_checkpoint
        self.finish_line=False
        self.index=index

    
    def radius_check(self, x_player, y_player):
        '''
        This function return whether or not the player is in the radius of the chekcpoint

        Parameters
        ----------
        x_player : TYPE float
            X player coordinate in the airsim coordinate system.
        y_player : TYPE float
            Y player coordinate

        Returns
        -------
        check : TYPE boolean

        '''
        
        return (x_player-self.x)**2+(y_player-self.y)**2<=self.r**2
    
    
    
class Circuit():
    def __init__(self, liste_checkpoints = []):
        self.active_checkpoints = liste_checkpoints
        
    def cycle_tick(self, x_player, y_player):
        '''
        Performs a regular cycle tick : checking player contact, updates the
        active chekpoints and return a boolean when a gate has just been passed,
        and another when a finish line checkpoint was crossed

        Parameters
        ----------
        x_player : TYPE float
            X player coordinate in the airsim coordinate system.
        y_player : TYPE float
            Y player coordinate

        Returns
        -------
        gate_passed : TYPE boolean
        end_race : TYPE boolean 

        '''
        if self.active_checkpoints==[]:
            raise TypeError("The circuit has no checkpoints to check")
        
        
        gate_passed = False
        end_race = False
        
        # Checking the proximity
        index_checkpoint = 0
        for checkpoint in self.active_checkpoints:
            if checkpoint.radius_check(x_player, y_player):
                gate_passed = True
                if checkpoint.next_checkpoint != None :
                    self.active_checkpoints[index_checkpoint] = checkpoint.next_checkpoint
                else:
                    self.active_checkpoints.pop(index_checkpoint)
                    index_checkpoint-=1
                if checkpoint.finish_line == True:
                    end_race = True
            index_checkpoint +=1
            
        return gate_passed, end_race


def circuit_fromlist(list_checkpoint_coordinates, spawn_point, loop=True):
    """
    Generates a circuit made of checkpoints, from a given list of UE coordinates.
    Very convenient when there are a lot of points. The input list has to go
    from the first to the last point in the racing order

    Parameters
    ----------
    list : TYPE list
        [ [x1, y1, r1] , [x2, y2, r2], ...] the coordinates are expected in UE coordinates.
        X1,Y1 will be the starting point, and X2, Y2 the second.
    spawn_point : TYPE numpy array
        The coordinates of the spawn_point (player start in UE)
    loop (optionnal) : TYPE boolean
        whether the circuit loops back on the first point or has an ending line.
    Returns
    -------
    Circuit : TYPE Circuit

    """
    xl,yl,rl=list_checkpoint_coordinates[-1][0], list_checkpoint_coordinates[-1][1], list_checkpoint_coordinates[-1][2]
    xl,yl, _ = convert_global_to_relative_position(spawn_point,np.array([xl, yl, 0]))
    last_checkpoint = Checkpoint(xl, yl, rl/2, index=len(list_checkpoint_coordinates)-1)
    previous_checkpoint = last_checkpoint
    for index_checkpoint in range(len(list_checkpoint_coordinates)-2,-1,-1):
        xi,yi,ri=list_checkpoint_coordinates[index_checkpoint][0], list_checkpoint_coordinates[index_checkpoint][1], list_checkpoint_coordinates[index_checkpoint][2]
        xi,yi, _ = convert_global_to_relative_position(spawn_point,np.array([xi, yi, 0]))
        checkpoint_i = Checkpoint(xi, yi, ri/2, previous_checkpoint, index=index_checkpoint)
        previous_checkpoint = checkpoint_i
        
    if loop:
        last_checkpoint.next_checkpoint = checkpoint_i
    else :
        last_checkpoint.finish_line = True
    
    circuit = Circuit([checkpoint_i])
    return circuit


        
    
    

 