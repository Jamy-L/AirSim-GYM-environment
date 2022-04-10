# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 19:37:46 2022

@author: jamyl
"""

from threading import Thread, Timer
import time
from queue import Queue
from stable_baselines3 import SAC
from jamys_toolkit import lidar_formater, denormalize_action


# _________ global variables ________________
TIME = time.time()
DECISION_PERIOD = 0.1  # seconds
OBSERVATION_PERIOD = 0.01
MODEL = SAC.load("P:/Final_benchmark/Training_V2/1004000")


# __________ Control functions _________________
def decision_making(obs):
    action = MODEL.predict(obs, deterministic=True)[0]
    return denormalize_action(action)


def push_action(action):
    pass


def fetch_observation():
    obs = None
    return obs


# ____________ MultiThreading stuff ____________________________


def decision_making_thread(obs_q):
    """ Choose an action based on observations

    Parameters
    ----------
    obs_q : Queue
        Queue containing the latest observation
    action_q : Queue
        Queue where the action will be passed

    Returns
    -------
    None.

    """
    # Calling back the same thread for periodic decision making
    Timer(DECISION_PERIOD, decision_making_thread, args=(obs_q,),).start()

    # getting the latest observation
    obs = obs_q.get()

    # deciding
    # print("Deciding", obs)
    action = decision_making(obs)

    # pushign action
    push_action(action)


def fetch_observation_thread(obs_q):
    """ Fetch the observation to feed the observation queue

    Parameters
    ----------
    obs_q : Queue
        queue containing the observation

    Returns
    -------
    None.

    """
    Timer(OBSERVATION_PERIOD, fetch_observation_thread, args=(obs_q,),).start()
    obs = fetch_observation()
    if obs_q.empty():  # Queue are FIFO. We are only using 1 element
        obs_q.put(obs)


# ______________ Thread initialisation and general stuffs _____________________
def main():

    # __________ Threads ___________________
    obs_q = Queue()

    decision_t = Thread(target=decision_making_thread, args=(obs_q,))
    obs_t = Thread(target=fetch_observation_thread, args=(obs_q,))

    decision_t.start()
    obs_t.start()


# _________ Starting main ______________________

main()
