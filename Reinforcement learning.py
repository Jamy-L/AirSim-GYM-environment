# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:13:35 2022

@author: jamyl
"""


from stable_baselines3 import SAC, PPO, DQN, TD3


import numpy as np
import airsim
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.save_util import save_to_pkl
import time
import random

import sys

sys.path.append("Training/Lidar_only")
from jamys_toolkit import (
    fetch_action,
    pre_train,
    create_spawn_points,
)
from jamys_toolkit import Jamys_CustomFeaturesExtractor
from Airsim_gym_env import (
    BoxAirSimEnv,
    MultiDiscreteAirSimEnv,
    DiscreteAirSimEnv,
    normalize_action,
    BoxAirSimEnv_5_memory,
    BoxAirSimEnv_MultiAgent,
)
from sim_to_real_library import lidar_sim_to_real

SAC.pre_train = pre_train  # Adding my personal touch


###############################################################################
# RC circuit model lidar only branch

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.simPause(True)
client.enableApiControl(True)

spawn = np.array([-13316.896484, 4559.699707, 322.200134])


liste_checkpoints_coordonnes = [
    [-10405.0, 4570.0, 10],
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
    [-14255.0, 4580.0, 10],
]


liste_spawn_point = create_spawn_points(spawn)

ClockSpeed = 4
airsim_env = BoxAirSimEnv_5_memory(
    client,
    dt=0.1,
    ClockSpeed=ClockSpeed,
    lidar_size=100,
    UE_spawn_point=spawn,
    liste_checkpoints_coordinates=liste_checkpoints_coordonnes,
    liste_spawn_point=liste_spawn_point,
    random_reverse=True,
)


models_dir = ""
logdir = ""
path = ""

TIMESTEPS = 1000


model = SAC.load(
    "C:/Users/jamyl/Desktop/Lidar_only.zip", tensorboard_log="P:/Training_V1"
)
obs = airsim_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = airsim_env.step(action)
    if done:
        obs = airsim_env.reset()

#%% training a model
model = SAC("MultiInputPolicy", airsim_env, verbose=1, tensorboard_log=logdir)

iters = 0
while True:
    iters = iters + 1
    model.learn(
        total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="Multi_agent",
    )
    model.save(f"{models_dir}/{TIMESTEPS*iters}")


# %% Trainign a model with custom extractor
models_dir = "P:/Training/Training_V4"
logdir = "P:/Training/Training_V4"


policy_kwargs = dict(
    features_extractor_class=Jamys_CustomFeaturesExtractor,
    features_extractor_kwargs=dict(
        Lidar_data_label=["current_lidar", "prev_lidar"], lidar_output_dim=100
    ),
)

TIMESTEPS = 1000

model = SAC(
    "MultiInputPolicy",
    airsim_env,
    verbose=1,
    tensorboard_log=logdir,
    policy_kwargs=policy_kwargs,
)


iters = 0
while True:
    iters = iters + 1
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name="SAC_Lidar_only_RC",
    )
    model.save(f"{models_dir}/{TIMESTEPS*iters}")


# %% Testing the model after learning

model = SAC.load("P:/Training_V1/504400", tensorboard_log="P:/Training_V1")
obs = airsim_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = airsim_env.step(action)
    if done:
        obs = airsim_env.reset()


# %% pretraining the model on prerecorded master trajectories

models_dir = "C:/Users/jamyl/Desktop/TER_dossier/Training"
logdir = "C:/Users/jamyl/Desktop/TER_dossier/Training"


TIMESTEPS = 100
model = SAC("MultiInputPolicy", airsim_env, verbose=1, tensorboard_log=logdir)

model.pre_train(replay_buffer_path=path)


# %% gathering teacher data to save in a replay buffer
# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.simPause(True)
client.enableApiControl(True)

spawn = np.array([-13316.896484, 4559.699707, 322.200134])


liste_checkpoints_coordonnes = [
    [-10405.0, 4570.0, 10],
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
    [-14255.0, 4580.0, 10],
]


liste_spawn_point = create_spawn_points(spawn)

ClockSpeed = 2
airsim_env = BoxAirSimEnv(
    client,
    dt=0.1,
    ClockSpeed=ClockSpeed,
    lidar_size=200,
    UE_spawn_point=spawn,
    liste_checkpoints_coordinates=liste_checkpoints_coordonnes,
    liste_spawn_point=liste_spawn_point,
    random_reverse=False,
)


models_dir = "P:/Benchmark/Training_V6"
logdir = "P:/Benchmark/Training_V6"
path = "P:/Replay_buffer"


playing_time = 300  # in seconds


airsim_env.reset()
airsim_env.close()


replay_buffer = DictReplayBuffer(
    buffer_size=1_000_000,
    observation_space=airsim_env.observation_space,
    action_space=airsim_env.action_space,
)

airsim_env.close()
client.simPause(True)
action = normalize_action(fetch_action(client))
observation, reward, done, info = airsim_env.step(np.array([0, 0]))
starting_time = time.time()
while True:
    action = normalize_action(fetch_action(client))
    future_observation, future_reward, future_done, info = airsim_env.step(
        np.array([0, 0])
    )
    replay_buffer.add(
        observation,
        next_obs=future_observation,
        action=action,
        reward=reward,
        done=done,
        infos=[info],
    )
    observation, reward, done = future_observation, future_reward, future_done
    if done:
        airsim_env.reset()

    if time.time() - starting_time >= playing_time:
        break

print("saving...")
save_to_pkl(path, replay_buffer)


# %% Load a previously trained model
model = SAC.load(
    "C:/Users/jamyl/Documents/GitHub/AirSim-GYM-environment/Training/32800",
    tensorboard_log="C:/Users/jamyl/Documents/GitHub/AirSim-GYM-environment/Training",
)


model.set_env(airsim_env)
airsim_env.reset()
airsim_env.render()

while True:
    model.learn(total_timesteps=1000, reset_num_timesteps=False, tb_log_name="SAC_1")


# %% Making random control

airsim_env.reset()
airsim_env.render()

for _ in range(10000):

    action = np.random.rand(3)
    # normalisation le steering doit être entre -0.5 et 0.5
    action[1] = action[1] - 0.5
    action[2] = 0  # pas de freinage

    if _ < 10:
        action[1] = 0.5
    airsim_env.step(action)


airsim_env.reset()
airsim_env.close()


# %% checking SB3 compatibility

check_env(airsim_env)
