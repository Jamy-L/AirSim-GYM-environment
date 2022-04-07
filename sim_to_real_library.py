# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:51:44 2022

@author: jamyl
"""

import numpy as np


def lidar_sim_to_real(sim_type_lidar):
    """
    Converts simulated lidar type into real lidar type

    Parameters
    ----------
    sim_type_lidar : np array([[theta_1, ..., \theta_n], [r_1, ..., r_n]])
        Lidar returned by airsim. theta is betwen -pi and +pi
        and the raidus is in airsim meters


    Returns
    -------
    real_type_lidar : TYPE TYPE np array([[theta_1, ..., \theta_n], [r_1, ..., r_n]])
        Lidar returned by the real lidar. theta is betwen 0 and 360° in growing
        order and the angle is in mm

    """
    scaling_factor = 6.25 / 1000
    real_type_lidar = np.copy(sim_type_lidar)

    # re organising the negative angles into theta >pi
    neg_angles = sim_type_lidar[sim_type_lidar[:, 0] <= 0][:, 0]
    real_type_lidar[:, 0][real_type_lidar[:, 0] <= 0] = 2 * np.pi + neg_angles
    # connverting rads to degrees
    real_type_lidar[:, 0] *= 180 / np.pi

    # scaling the distance
    real_type_lidar[:, 1] /= scaling_factor

    # applying a mirroring on angles
    real_type_lidar[:, 0] = 360 - real_type_lidar[:, 0]

    # sorting by theta to get the right order
    idx = real_type_lidar[:, 0].argsort()
    real_type_lidar = real_type_lidar[idx, :]
    return real_type_lidar


def lidar_real_to_sim(real_type_lidar):
    """
    Converts real lidar type into airsim lidar type

    Parameters
    ----------
    real_type_lidar :  np array([[theta_1, ..., \theta_n], [r_1, ..., r_n]])
        Lidar returned by the real lidar. theta is betwen 0 and 360° and the angle is in mm

    Returns
    -------
    sim_type_lidar : TYPE np array([[theta_1, ..., \theta_n], [r_1, ..., r_n]])
        Lidar returned by airsim. theta is betwen -pi and +pi and the radius is in airsim meters
    """
    scaling_factor = 6.25 / 1000
    sim_type_lidar = np.copy(real_type_lidar)
    # reorganising the theta>180 into theta<0

    neg_angles = real_type_lidar[real_type_lidar[:, 0] >= 180][:, 0]
    sim_type_lidar[:, 0][sim_type_lidar[:, 0] >= 180] = neg_angles - 360

    # degrees to rads
    sim_type_lidar[:, 0] *= np.pi / 180

    # scaling the distance
    sim_type_lidar[:, 1] *= scaling_factor

    # mirroring the angles
    sim_type_lidar[:, 0] *= -1
    # sorting by theta to get the right order
    idx = sim_type_lidar[:, 0].argsort()
    sim_type_lidar = sim_type_lidar[idx, :]

    return sim_type_lidar
