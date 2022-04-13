# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 19:37:46 2022

@author: jamyl
"""
import numpy as np
from threading import Thread, Timer
import time
from queue import Queue
from stable_baselines3 import SAC
from rplidar import RPLidar
import sim_to_real_library as s2r
from pynput import keyboard
import spidev
from jamys_toolkit import lidar_formater, denormalize_action


# _________ global variables ________________
TIME = time.time()
DECISION_PERIOD = 0.1  # seconds
OBSERVATION_PERIOD = 0.1
MODEL = SAC.load("/home/pi/Documents/vroom/Lidar_only")

# ________ States ______________
STATE = None
STOP = 0
DRIVING = 1
EVASIVE_MANEUVER = 2
# ________ LIDAR DEFINITION _________________
PORT_NAME = "/dev/ttyUSB0"

LIDAR = RPLidar(PORT_NAME)


def start_lidar():
    try:
        return LIDAR.get_info()
    except:
        start_lidar()


info = start_lidar()

print(info, type(info))

health = LIDAR.get_health()
print(health)


ITERATOR = LIDAR.iter_scans()  # Getting scan (type = generator)
OBSERVATION = None

# ____________ SPI DEFINITION ____________
BUS = 0
DEVICE = 0

SPI = spidev.SpiDev()
SPI.open(BUS, DEVICE)

SPI.max_speed_hz = 1000000
SPI.mode = 0

# __________ Control functions _________________
def sat(x, xmin, xmax):
    """ saturation function

    Parameters
    ----------
    x : float
        Input number
    xmin : float
        low saturation bound
    xmax : float
        high saturation bound

    Returns
    -------
    float
        Satured version of x

    """
    if x > xmax:
        return xmax
    if x < xmin:
        return xmin
    return x


def evasive_maneuver(obs):
    global STATE
    action = None

    drive_condition = False
    if drive_condition:
        STATE = DRIVING
    return action


def car_ctrl(steering, motor_speed):

    # Commands limits
    motor_speed = sat(motor_speed, xmin=0, xmax=10)

    steering = sat(steering, xmin=-15, xmax=15)

    # Sending commands
    to_send = [motor_speed, steering + 135]
    reply = SPI.xfer2(to_send)


def decision_making(obs):
    if STATE == DRIVING:
        action = denormalize_action(MODEL.predict(obs, deterministic=True)[0])

    elif STATE == STOP:
        action = np.array([0, 0])  # 0 speed, 0 steering

    elif STATE == EVASIVE_MANEUVER:
        action = evasive_maneuver(obs)
    else:
        raise ValueError(
            """ STATE took unexpected value. Expected {}, {}
                         or {}, but received {}""".format(
                DRIVING, STOP, EVASIVE_MANEUVER, STATE
            )
        )
    return action


def push_action(action):
    """
    action[0] : 'throttle' -> [-1, 1]
    action[1] : steering -> [-0.5, 0.5]

    """
    print(action)
    throttle = 10 * abs(action[0])
    steering = 30 * action[1]

    car_ctrl(steering, throttle)


def fetch_observation():
    global STATE
    stuck_condition = False
    if stuck_condition:
        STATE = EVASIVE_MANEUVER

    scan = next(ITERATOR)
    dots = np.array([(meas[1], meas[2]) for meas in scan])  # Convert data into np.array

    current_lidar, conversion_error = lidar_formater(s2r.lidar_real_to_sim(dots), 100)
    if conversion_error:
        raise ValueError("No lidar data was received")

    global OBSERVATION
    if OBSERVATION is None:  # init
        OBSERVATION = {
            "current_lidar": current_lidar,
            "prev_lidar1": np.copy(current_lidar),
            "prev_lidar2": np.copy(current_lidar),
            "prev_lidar3": np.copy(current_lidar),
            "prev_lidar4": np.copy(current_lidar),
            "prev_lidar5": np.copy(current_lidar),
            "prev_throttle": np.array([0]),
            "prev_steering": np.array([0]),
        }
    else:
        prev_lidar5 = OBSERVATION["prev_lidar4"]
        prev_lidar4 = OBSERVATION["prev_lidar3"]
        prev_lidar3 = OBSERVATION["prev_lidar2"]
        prev_lidar2 = OBSERVATION["prev_lidar1"]
        prev_lidar1 = OBSERVATION["current_lidar"]

        OBSERVATION = {
            "current_lidar": current_lidar,
            "prev_lidar1": prev_lidar1,
            "prev_lidar2": prev_lidar2,
            "prev_lidar3": prev_lidar3,
            "prev_lidar4": prev_lidar4,
            "prev_lidar5": prev_lidar5,
            "prev_throttle": np.array([0]),
            "prev_steering": np.array([0]),
        }

    return OBSERVATION


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


def on_press(key):
    global STATE
    try:
        if key.char == "p":  # pause"
            STATE = STOP
            print("PAUSE")

        elif key.char == "s":  # start
            STATE = DRIVING
            print("STARTING...")

    except AttributeError:
        print("special key {0} pressed".format(key))


# ______________ Thread initialisation and general stuffs _____________________
def main():
    # ______ Keyboard init ______________________
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # __________ Threads ___________________
    obs_q = Queue()

    decision_t = Thread(target=decision_making_thread, args=(obs_q,))
    obs_t = Thread(target=fetch_observation_thread, args=(obs_q,))

    decision_t.start()
    obs_t.start()


# _________ Starting main ______________________

main()
