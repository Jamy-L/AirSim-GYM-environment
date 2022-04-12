# Autonomous Vehicles using AirSim, Lidar and Vision
# Written in 2021-2022 by Rida Lali, Jamy Lafenetre
# Pierre-Alexandre Peyronnet and Kevin Zhou

"""Fonctions pour l'utilisation du Lidar"""

import numpy as np
import random
from rplidar import RPLidar, RPLidarException

"""Lidar Definition"""
PORT_NAME = '/dev/ttyUSB0'
lidar = RPLidar(PORT_NAME)

"""Constantes"""
detection_angle_min = 45
detection_angle_max = 90

"""Fonctions"""
def find_detection_angle(front_dist):
    """
    Trouve le bon angle de detection selon la distance devant la voiture

    Parameters
    ----------
        front_dist : float
            Distance devant

    Return
    ------
        float
            Angle de detection
    """
    detection_angle = -angle_coeff * (front_dist - 600) + detection_angle_max
    return min(max(detection_angle,detection_angle_min), detection_angle_max)

def get_scans():
    """
    Obtenir les iterateurs de scans

    Return
    ------
        iterator
            Table des mesures
    """
    return lidar.iter_scans()

def traitement_scan(iter):
    """
    Traitement du scan

    Parameters
    ----------
        iter    :  iterator
            Avoir le bout du scan

    Return
    ------
        front_dists, left_dists, right_dists
        front_dist, left_dist, right_dist
    """
    scan = next(iter)
    dots = np.array([(meas[1], meas[2]) for meas in scan])

    #Find front distance
    front_dists = np.concatenate((dots[:,1][dots[:,0] < 10],dots[:,1][dots[:,0] > 350]))
    front_dist = np.mean(front_dists)

    #Find detection angle
    detection_angle = find_detection_angle(front_dist)

    #Find distance on the left
    left_dists = dots[:,:][detection_angle+5 > dots[:,0]]
    left_dists = left_dists[:,1][left_dists[:,1] > detection_angle-5]
    left_dist = np.mean(left_dists)

    #Find distance on the right
    right_dists = dots[:,:][dots[:,0] > 355-detection_angle ]
    right_dists = right_dists[:,1][right_dists[:,0] < 365-detection_angle ]
    right_dist = np.mean(right_dists)

    return (front_dists, left_dists, right_dists)
