# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 08:50:14 2022

@author: jamyl
"""

import numpy as np
import random
import gym
import torch as th
from torch import nn

from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import  SACPolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)



from gym import spaces
from torch.nn import functional as F

######### Policy stuff #########################################################
def preprocess_lidar_tensor(obs: th.Tensor):
    '''
    Preprocess specifically lidar data OF THE FORMAT [THETA, R]

    Parameters
    ----------
    obs : th.Tensor
        DESCRIPTION.

    Returns
    -------
    None.

    '''


def preprocess_obs(
    obs: th.Tensor,
    observation_space: spaces.Space,
    normalize_images: bool = True,
) -> Union[th.Tensor, Dict[str, th.Tensor]]:
    """
    Preprocess observation to be fed to a neural network.
    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
    For discrete observations, it create a one hot vector. Lidar data are treated by convolution

    :param obs: Observation
    :param observation_space:
    :param normalize_images: Whether to normalize images or not
        (True by default)
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        if is_image_space(observation_space) and normalize_images:
            return obs.float() / 255.0
        return obs.float()
    


    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return F.one_hot(obs.long(), num_classes=observation_space.n).float()

    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        return th.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
                for idx, obs_ in enumerate(th.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.float()

    elif isinstance(observation_space, spaces.Dict):
        # Do not modify by reference the original observation
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs(_obs, observation_space[key], normalize_images=normalize_images)
        return preprocessed_obs

    else:
        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")
        
        
def extract_features(self, obs: th.Tensor) -> th.Tensor:
    """
    Preprocess the observation if needed and extract features.

    :param obs:
    :return:
    """
    assert self.features_extractor is not None, "No features extractor was set"
    preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
    return self.features_extractor(preprocessed_obs)

# Overwritting the extract feature method because we will use a custom a different 
# preprocessing, normalizing lidar data.

BaseModel.extract_features = extract_features 



class Jamys_CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    A Custom Feature Extractor working with Dict observation Spaces. Images are
    treated like others featur extractors would : 3 convolutions layers. Lidar
    data are passed through multiple layers of convolutions, and other types
    of input are simply passed through a fltten layer. At the end, all features
    are concatenated together

    :param observation_space:
    :param Lidar_data_label: List of the observations name of lidar data
    :param lidar_output_dim: Number of features to output from each CNN submodule(s) dedicated to Lidar. Defaults to
        100 to avoid exploding network sizes.
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: gym.spaces.Dict,Lidar_data_label=[], lidar_output_dim: int = 100, cnn_output_dim: int = 256):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        # I dunno man, I think I can live with that for now ...
        super(Jamys_CustomFeaturesExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}
        Lidar_extractor = None #init, all Lidar channels will share the same NN
        def create_Lidar_extractor(subspace, lidar_output_dim):
            Lidar_extractor = LidarCNN(subspace, features_dim = lidar_output_dim, kernel_height=5)
            return Lidar_extractor

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            elif key in Lidar_data_label :
                if Lidar_extractor == None:
                    Lidar_extractor = create_Lidar_extractor(subspace, lidar_output_dim)
                extractors[key] == Lidar_extractor
                total_concat_size += lidar_output_dim
                
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)
    
    
class Jamys_CustomPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(Jamys_CustomPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )
        
     
class LidarCNN(BaseFeaturesExtractor): #TODO
    """


    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, kernel_height = 5):
        # TODO is normalisation mandatory ?
        super(LidarCNN, self).__init__(observation_space, features_dim)
        # We assume On channel, H x 2 X 1 format
        # Re-ordering will be done by pre-preprocessing or wrapper
        # assert is_image_space(observation_space, check_channels=False), (
        #     "You should use NatureCNN "
        #     f"only with images not with {observation_space}\n"
        #     "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
        #     "If you are using a custom environment,\n"
        #     "please check it using our env checker:\n"
        #     "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        # )
        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=(kernel_height, 2), stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(0, 1),
            nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    
##### Pre-training and teacher demonstration ################################### 
def pre_train(self, replay_buffer_path, gradient_steps = 1000):
    '''
    Pretrains the model based on a pre-recorded replay buffer : a demonstration
    of a teacher policy for example

    Parameters
    ----------
    replay_buffer_path : TYPE string
        local path of a pre-recorded replay buffer (only DictReplayBuffer type
        are supported so far, so nothing like HERReplayBuffer are supported )

    Returns
    -------
    None.

    '''
    self.load_replay_buffer(replay_buffer_path)
    self.replay_buffer.device='cuda' # TODO is it really working ??
    
    self._setup_learn(total_timesteps = 10, eval_env=None)
    print("training the model from teacher's demonstration")
    self.train(gradient_steps=gradient_steps)
    print('end of training')

################ Data related ##################################################
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


def fetch_action(client):
    '''
    Returns the vehicule command performed by the human user, on an MDP format

    Parameters
    ----------
    client : TYPE AirSim client
        AirSim client

    Returns
    -------
    TYPE numpy array
        ([throttle, steering])

    '''
    controls = client.getCarControls()
    return np.array([controls.throttle , controls.steering])

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

########## Checkpoints and spawn #############################################
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
        This function return whether or not the player is in the radius of the checkpoint

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

class Circuit_wrapper():
    def __init__(self,circuit_spawn_list, list_checkpoint_coordinates, UE_spawn_point):
        self.spawn_point_list = circuit_spawn_list
        self.list_checkpoint_coordinates = list_checkpoint_coordinates
        self.UE_spawn_point=UE_spawn_point
        
    def sample_random_spawn_point(self):
        self.selected_spawn_point = random.choice(self.spawn_point_list)
        self.theta_spawn = random.uniform(self.selected_spawn_point.theta_min, self.selected_spawn_point.theta_max)
        
        self.generate_circuit(self.selected_spawn_point)
        
        return self.selected_spawn_point , self.theta_spawn, self.circuit
    
    
    def generate_circuit(self, selected_spawn_point):
        liste_checkpoints=self.list_checkpoint_coordinates
        index_recalage = selected_spawn_point.checkpoint_index
        liste_checkpoints = liste_checkpoints[index_recalage:]+liste_checkpoints[0:index_recalage] #recalage de l'ordre des checkpoints
        self.circuit = circuit_fromlist(liste_checkpoints, self.UE_spawn_point)
        
    
    
class Circuit_spawn():
    def __init__(self, x, y, z, theta_min, theta_max, checkpoint_index, spawn_point):
        '''
        

        Parameters
        ----------
        x : TYPE float
            position of the circuit spawn point given by UE.
        y : TYPE float
            
        z : TYPE float
           
        teta_min : float
            in radians, minimum angle of deviation when spawning randomly at this point
        teta_max : float
            maximum angle
        checkpoint_index : int
            index of the first checkpoint that must be crossed. The index is
            relative to the list original fed to circuit_fromlist()
        spawn_point : TYPE numpy array
            coordinates of the player spawn in UE

        Returns
        -------
        None.

        '''
        x,y,z = convert_global_to_relative_position(spawn_point, np.array([x,y,z]))
        self.x=x
        self.y=y
        self.z=z
        
        self.theta_min=theta_min
        self.theta_max=theta_max
        
        self.checkpoint_index=checkpoint_index
        
        
        
        
        

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


        
    
    

 