# AirSim GYM environment
This set of programs aims to make reinforcement learning on AirSim easier, by providing a GYM custom environment. It is fully compatible with Stable Baselines 3 which I recommend using.

## Beginning
Make sure to be familiar with <a href="https://microsoft.github.io/AirSim/" target="_blank">AirSim</a>. It requires Unreal Engine 4 and some python modules like open CV. Cuda is also strongly recommended, and you may want to use a beefy GPU with at least 16Go of RAM to be serein.

You should also check <a href="https://gym.openai.com/" target="_blank">GYM</a> which is a python environment library that basically makes reinforcement learning much easier to program.

<a href="https://stable-baselines3.readthedocs.io/en/master/" target="_blank">Stable Baselines 3</a> provides a set of very performants and modulable RL algorithms and is based on PyTorch. Make sure to have a clean install of the latter.

## Presentation
A GYM environment is basically composed of 3 methods: 
<ol>
 <li> <code>step()</code> takes an action input and performs it on the system.
      It is then returning an observation following the action, a computed reward estimated for the timestep, a boolean indicating whether or not the episode should be finished, and some debugging infos. In this project, the time step is configurable as a given time to elapse before further decision making. Please read the dedicated section for more information on this choice</li>
 <li> <code>reset()</code> basically acts as an initialisation, and is called everytime a simulation ends. That's where all the randomness occurs, to avoid overfitting. It has to return an observation too.</li>
 <li> <code>render()</code> is supposed to be (maybe) called once and to render the simulation each step. Here it's really not that useful since it will run anyway in Unreal Engine, but I have made a little plot of the lidar and camera, although the optimization is terrible (sorry) 😥</il>
 <li> <code> close()</code> is supposed to close the system. In this program it just gives back the control to the player, so it's a great way to make sure a player can take control. </li>
 <li> <code> spawn_points_checker()</code> Was introduced by me. It just reviews all the spawn points and tests the extreme values of spawn angles for each, so you can visually check if there are no mistake.
</ol>

### Observations and actions
Observation and actions spaces have to be specified while initiating the environment in <code> __init__</code>. A lower and upper bound have to be predefined (you can put +infinite so it's not too restrictive), and the types have to be mentioned. It's generally a good idea to simulate a first order memory by adding the previous step's action and observations as an observation.

I propose to take AirSim's lidar data as an observation vector. The SETTINGS.JSON file contains all the settings related to that (number of points, speed, angle, channel number ...). Keep in mind that the lidar data's size is changing at each step depending on what the lidar is hitting... To that I propose a very simple fix: if there are too many points: crop them. If there are too little points: make a padding and just recopy one of the points. Also, <code>jamyys_toolkit.py</code> contains a function to convert the cartesian coordinates to polar. Just make sure to set global coordinates to relative coordinates in SETTINGS.JSON.

Camera feed is also a good observation source, but it's very expensive.




Actions basically consists of the throttle and the steering, nothing more.

## Reward function
### Checkpoints and spawns
I have implemented a random respawn and checkpoint system. Just add collision-free objects in Unreal Engine (like transparent Spheres) to mark the checkpoints, and add their coordinates in the program. You also need to give the coordinates of the "player spawn" object in the UE project because the coordinates given by the AirSim API are relative to it. Make sure to give them in the right order !. The <code> Circuit_wrapper</code> class of <code> jamys_toolkit.py</code> will directly make all the conversion, and make a circular infinite loop.

For respawn points, you can make the same thing and specify a minimum and maximal deviation angle (when spawning, a random value will be sampled between the two). You also need to manually specify what is the index of the first checkpoint to validate, since it depends on where the car is spawning.

 ### Reward
 Reward is calculated as: 
 <ul>
 <li> -100 points in case of crash, and the episode is ending </li>
 <li> +50 points when a checkpoint is validated </li>
 <li> -0.1*speed points when the speed is below is under 1 m/s (no slowpoke on the racing track !) </li>
 </ul>
  "going fast" is implicitly a synonym of marking a lot points, since the episode time is capped at 1000 steps (reached if there is no crash).
  
  ## About time steps and simulation speed
  This is a very though matter. Simulating a Markov decision process generally requires to chose a time step separating two actions. Typically, GYM scheme is "fetch observation"-"choose action"-"calling a simulation step". To simulate this scheme, my method is to have the simulation paused at all time, except for the specific window where <code> step()</code> is called. In itself, this method has it flaws because real environment will not pause when calculations are made. However, the delays involved by inference are neglectable with regards to the delays involved with training a network. I believe the key element is to ensure that observations (both for a simulated and real environment) are fetched with regular time steps, because it enables some sort of speed estimation based on a first order memory of space observation. This behaviour is achievable on a real system thanks to onboard OS and proper task scheduling.
  
### What time is it?
  However, that being said, the time step problem is far from being solved. When speaking of clocks/frequencies/framerate/speed, many protagonists come to action.
  <ul>
 <li> Unreal Engine is a video games engine that need to generate a certain number of frames per second. The FPS are dependent on the hardware running the environment, and on the system's different active tasks. The FPS are not only a visual asset, they also intervene in the physics engine, and must be kept around a reasonable value. Typically, staying over 30 FPS is a must, and the maximum you can reach in the editor (that is without launching the game) is 120. </li>
 <li> PhysX is taking care of the vehicle's physics, by applying a finite difference scheme over time. This time step in particular is not easily reachable, but is tremendously important for simulating a realistic behaviour. For that reason, we have to ensure that this time step is not too big, because it would significantly worsen realism. For example, collisions would not be properly detected. I have also experimentally observed that when this time step gets too big, the dynamic behaviour changes a lot. A constant acceleration on a straight line would not get the car as far as it would with a shorter time step, although they are driving for the same duration with the same speed. I will delve further on this issue later </li>
 <li> AirSim also has its inner clock. It can be scaled in SETTINGS.JSON, which is really handy to speed up training by speeding simulations. </li>
 <li> The computer running the simulation itself has its own real time clock. <code> time.sleep()</code> enables to wait for a given real time between samples</li>
 </ul>
 
### So which clock should be used?
 A first idea is to use AirSim API's <code>simContinueForFrames()</code> or <code>simContinueForTime()</code>. However, I have found that this approach is not that good. First of all, a "frame step" doesn't really make sense in my opinion because it's very hard to transpose on a real environment. Furthermore, framerate drops due to external activities must be taken into account, and it impossible to predict how the framerate will react when you are doubling the simulation clock.
 
 <code>simContinueForTime()</code> Sounds good, but is a trap in my opinion. It is very inconsistent with short time steps, and clock scaling factors. This feature is discussed among the AirSim community, so I have decided to completely abandon it.
 
 My conclusion is that <code> time.sleep()</code> is the stepping method to use (you can of course use a dedicated thread instead, if you find it useful). Let's say for now that the time scaling factor is 1. My main argument is that Unreal Engine is meant to simulate real time environments, with framerates above 30 FPS. Therefore, in that scenario, 1s in real life / on the processor clock accounts for 1s in the simulation. The FPS are dynamically adjusting to ensure that the simulation is indeed real time, although this objective cannot be achieved if the FPS are dropping below 30 FPS (in that case the simulation clock will become slower than the real world clock). When the scaling factor is different than 1 (let's say it is bigger), physX needs to compute more time steps during a given time. We may talk of "effective framerate" equal to apparent framerate divided by clock scaling factor (FPS/Scale). For example, if my computer runs AirSim at 40 FPS, and I set a scaling factor of 4, it means that the effective FPS will be 10, which is much below 30FPS. In that case, PhysX time step will not be precise enough, and the physics will not be precise at all.
 
 ### How to speed the collect of trajectories ?
 By default, AirSim simulates a 1x1 time scale. For bovious reasons, you want to speed up the simulated time to get more data more quickly. The most logical way would be to set a PhysX delta time that is the biggest possible, while maintaining a coherent and realistic behaviour (by trial an error), and to let AirSim work with that. It would mean that the ClockSpeed is dynamic, and given by how fast your hardware can compute a time step. So if you are running it on a game boy color, the time would appear as slow down, whereas if you run it on an H100 it would go 100 times faster than real life. 
 
 Sadly this approach is not easily implementable in our case, mostly because of the choices made for Unreal Engine. Basically, the time step is choosen dynamically depending on the FPS, to ensure a 1x1 Clock speed, which is understandable. In our case, running Airsim on a game boy color would make a real time simulation with a terrible delta time, meaning that we could probably run through walls. On the contrary, on a H100 the simulation would be real time but very smooth and accurate.
 In reality, Unreal Engine's delta time is semi-dynamic, meaning it's semi-bounded. It can become as little as possible, but a maximum value has to be given. If the FPS drop too much, the delta time becomes bigger, but is capped at a value. It mean that in reality, AirSim on a game boy color would achieve <1 clock speed, and you could not get through walls.
 
 Another functionnality I have tested is to cap the FPS at a value (in the general settings of UE4). It means that if your FPS are above the objective, the hardware will stop computing further until the next step is required. It's very known in video games, because it can prevend hardware from overheating too much. In our case it's not that useful because we would rather have the hardware go above 1 clock speed. If the FPS are below the objective, The clock speed goes <1 and the slowed down simulation will appear smooth.
 
 All of this concludes what I have to say about AirSim and time steps. My final word on that is that so far, I suggest you to launch the project in a very low resolution and quality to get more FPS (there are a few guides on youtube on how to make a widget to show them on the screen). Based on the FPS, apply a product rule to choose a Clock Speed factor that can maintain FPS/scale >30 FPS or 40 FPS. Make sure NOT to cap the FPS.
 If you are brave you can try to manually change how all of this is working. I have also heard that Unity can work with fixed delta time, so you could try AirSim on unity.
 
 # My implementation of reinforcement learning methods to autonomous driving
 
 The AirSim environment uses a continuous action space and a continuous observation space. For clarity, the observations are of type "Multi Input" since lidar data, previous actions, and a picture are fetched every step. The action space is just a 2D vector containing steering and throttle, therefore it is multi input.
 
 The continuity of the problems narrows the choice of RL algorithms. Furthermore, due to the nature of the environment, observation data will be very scarce, which gives off-policy algorithms a lead against on-policy algorithms.
 
 These details being cleared, SAC and TD3 appear to be the most logical choice. I have personally chosen to use SAC because I was more familiar with it, but most of what will be said later can be generalized to other algotirhms working with continuous state-action spaces.
 
 ## Workings with SAC
 SAC is explained in more details here: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
 It is strongly recommended to read the original article to understand the inner core of the algorithm.
 
 In a pragmatic way, three protagonists come to action during the training: 
 <ul>
 <li> A critic network learns to predict the expected reward for any action, in a given state. There are actually 2 critic networks being trained at each step, and the best one is selected every time.</li>
 <li> The entropy coefficient basically regulates whether policies will rather explore or exploit the environment. It is evolving during the training, and helps a long to stabilize the learning process. Generally, the coefficient starts at a high value, and converges toward 0 at the end of training.</li>
 
 <li> The actor network maps a probability to every action for each given state </li>
 </ul>
 
 All these protagonists evolve at the same time, their training is tightly linked. The trainings are of course based on collected trajectories, on which the training is done. After training, a policy is deployed on the environment to collect more trajectories (it may be trajectories of exploration or exploitation), and the process cycles. These two processes are called <code>train</code> and <code>collect_rollout</code> (read more here: https://stable-baselines3.readthedocs.io/en/master/guide/developer.html)
 
 ![sb3_loop](https://user-images.githubusercontent.com/46826148/159230016-5d6be5a9-44b3-478c-bf0e-5c147db47886.png)
 
 
 
 ### Feature extraction
 What is not crystal clear at first sight is the link between "state" and "observation". An observation is in our case a dictionary, containing sets of observations of different types, while a state must be a 1d array which will be fed to the critic and the actor networks. A pre-processing layer called <code>feature_extractor</code> is actually transforming the observation to an estimated state. Stable Baselines uses by default a shared feature extractor for the actor and the critic, which makes sense in our case. (You can read more about that here: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
 
 ![net_arch](https://user-images.githubusercontent.com/46826148/159230138-622e2266-8ae9-408e-b4e4-bd9505c970c3.png)
 
 
 By default, Stable Baselines simply iterates through all the observations keys and makes the following :
 <ul>
 <li> If the observation is an image, a normalisation is done and 3 layers of convolutions are applied, followed by a linear transformation to make sure the number of features extracted from the image are right. (For more details on activation function, kernel size and stride, refer to the original code)</li>
 <li> If the observation is not an image, a <code>Flatten</code> layer is simply applied. If you are not familiar with PyTorch's terminology, it just means that all your columns are concatenated one after the other to form a vector.</li>
 </ul>

 The extracted features for every observation key are then once again concatenated into a single big 1D vector.
 
 This feature extractor is in my opinion sub-optimal, especially for treating lidar data. In our case, a lidar data is a 2 by N array, here N represents the number of lidar point collected. Therefore, a simple <code>Flatten</code> layer cannot extract feature with accuracy.
 
 I propose to use a couple of convolution layers on the lidar data, as the relation between each successive radius measure seems to carry precious information, independently from the global coordinate in the array. For example, the first point appearing in the observation array (r1, th1) changes of meaning at every observation, since th1 is never the same. Therefore, a fully connected network/Multi layer perceptron is bound to perform a bad feature extraction.
 On the contrary, studying a relation between r1, r2, r3,... can lead to detect a wall, a corner or a "hole". A convolution filter will sum ponderations of the ri, and may for example develop sets of spatially auto-regressive models: One for detecting corner, one for straight lines. A big gap between and ri and his auto-regressive predicted value may indicate that there is a gap for this given point.
 
 This is just an example of why I believe a convolution layer makes sense. We may also imagine that lidar data could in our case translate to a 2D map from the sky view, taking two values: one for the absence of wall, and another when a wall is detected. It would then make sense to use a convolution layer on such an image and I have no doubt that significant features would be extracted (although such a method would not be the most efficient).
 
 In practice, nothing such has to be implemented, simply choosing the hyper-parameters of the filter, like stride, padding, size and activation.
 
 
 
 
 Do bear in mind that in our case, two lidar data are observed:  The current one and the previous one, to simulate a first order memory. Obviously, the feature extractor for the current lidar should be the same as the previous lidar feature extractor (by that I mean that they should share the same parameters/weights). We can thus train a single lidar feature extractor.
 
 ### On the technial side of feature extraction
 Stable Baselines 3 uses 3 different "layers" for processing an observation :
 <ul>
 <li> Preprocessing applies a normalisation for images (from [0,255] to [0,1]), and applies casting, changes of format, conversion to torch tensors etc... </li>
 <li> The Feature Extractors works with the pre-processed observation tensors, and returns a Tensor representig the state of the Markovian decision process</li>
 <li> The rest of whatever you are doing, for example feed the actor network with the current estimated state to get a probability distribution over all action</li>
 </ul>
 
 Despite being seemingly a very dull task, preprocessing plays a tremendously important role in the algorithm. In our case, lidar data is a specific type that needs to be preprocessed separately from the rest, because it doesn't look like any sort of image, in the sense that r can vary from 0 to the infinite (There is maximum range reachable by the lidar, but you get the idea), and theta can vary from 0 to 360°. Instead of manually working with scaled and centered observation, the most logical way of doing is to modify the preprocessing layer to recognize and treat Lidar data accordingly.
 
 Check <code>common/policies.py</code>. It includes the <code>BaseModel</code> class, from which the majority of RL algorithms inherit. It has a method called <code> extract_features()</code>? which is called whenever an observation is pushed through the network (if you are lost, I suggest to try a <code>model.predict()</code> with debugger on, to see where it is leading. You will ultimately arrive there).
 
 ```python
def extract_features(self, obs: th.Tensor) -> th.Tensor:
   """
   Preprocess the observation if needed and extract features.

   :param obs:
   :return:
   """
   assert self.features_extractor is not None, "No features extractor was set"
   preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
   return self.features_extractor(preprocessed_obs)
 ```
 
 The terminology of Stable Baselines can be a bit confusing because as you can see, preprocessing seems included in the ferature extraction process.
 Let's look closer at the latter : 
 
 ```python
 def preprocess_obs(
    obs: th.Tensor,
    observation_space: spaces.Space,
    normalize_images: bool = True,
) -> Union[th.Tensor, Dict[str, th.Tensor]]:
    """
    Preprocess observation to be to a neural network.
    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
    For discrete observations, it create a one hot vector.

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
 ```
There we have it, the default preprocessing layer of Stable Baselines 3. How cool !
You can see that for Dict observation spaces like we have, it simply iterates through all the obseravtion sub-spaces. For each it just checks wheteher it's an image or not... which is done this way:

```python
def is_image_space(
    observation_space: spaces.Space,
    check_channels: bool = False,
) -> bool:
    """
    Check if a observation space has the shape, limits and dtype
    of a valid image.
    The check is conservative, so that it returns False if there is a doubt.

    Valid images: RGB, RGBD, GrayScale with values in [0, 255]

    :param observation_space:
    :param check_channels: Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected.
    :return:
    """
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if observation_space.dtype != np.uint8:
            return False

        # Check the value range
        if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
            return False

        # Skip channels check
        if not check_channels:
            return True
        # Check the number of channels
        if is_image_space_channels_first(observation_space):
            n_channels = observation_space.shape[0]
        else:
            n_channels = observation_space.shape[-1]
        # RGB, RGBD, GrayScale
        return n_channels in [1, 3, 4]
    return False
   ```
   It is simply a check of the lowest and highest value, along with a dimension check. My initiative is to rewrite these parts, to include the lidar data type and make sure on no account it will ever be lumped with an image. To do that, <code>jamys_toolkit</code> overwrites the <code>extract_features</code> method of the <code>BaseModel</code>. By doing so, the original code is not modified and everything works fine. The trick is simply to redefine <code>preprocess_obs</code>.
 
 ### Implementation of my custom preprocessing layer and feature extractor
 Preprocessing simply consists off a normalisation of the radius and angle. The angle is "simply" divided by 2 times pi, and the radius is noramlized by the function y = -exp(-x) + 1. It is not the only option available of course, and maybe not the best, but I think it makes the job perfectly. You should be aware that so called "in-place operation" must be avoided at all cost inside any layer. In-line operation basically means wrtitting over a variable value, so A[:,:,0]+= 1 is forbidden for example. If you do so, backtracking the gradient will be impossible and the debugger will give you a mysterious message whishing you good luck for debugging 😟... 
 
It can be a bit tricky to reshape and normalize tensors without that, so here is my pre-processing layer :
```python
def preprocess_lidar_tensor(obs1: th.Tensor, observation_space: spaces.Space,):
    '''
    Preprocess specifically lidar data OF THE FORMAT [[THETA1, R1], ..., [THETA_N, R_N]]
    it includes normalidation and casting to float

    Parameters
    ----------
    obs : th.Tensor
        Observation of the lidar format. Please make sure that the observation
        space is a Box type.

    Returns
    -------
    preprocessed obs : A normalized tensor

    '''
    if not isinstance(observation_space, spaces.Box):
        raise TypeError("The observation space is not a box ....")
    else :
        obs = obs1.float()
        c0 = obs[:,:,0, None]/2*np.pi
        c1 = - th.exp(-obs[:,:,1, None])+1 #passing from [0, infty[ to [0, 1]
        normalized = th.cat((c0,c1), dim =2)
        obs2 = normalized[:,None,:,:] # adding a fourth dimension

        # This reshapes is essential, since the function receives a 3d tensor.
        # The first dimension receives the batch size. This reshape allow to
        # keep the batch size on the 0 dimension, then channel on 1, regular
        # Lidar shape on 2, 3
        return obs2

```

When it comes to the feature extractor, I have introduced a "Concat layer" which concatenates the differents channels into a single 2D images.
Once again, you will see that no variable is overwritten.

``` python
class Concat(nn.Module):
    def __init__(self, ):
        super(Concat, self).__init__()

    def forward(self, x):
        x_new = x.clone()
        original_shape = x_new.shape
        batch_size = original_shape[0]
        channels = original_shape[1]
        
        return x_new.reshape(batch_size,1, original_shape[2], channels)
```
 ### Action normalisation
 When checking your custom environment with the <code>check_env()</code> function of stable baselines 3 (which is by the way very useful, thank you for implementing that !), you may be welcomed with a warning message saying that your observations and actions are not normalized, which may be a problem. We have already talked of the normalisation of observation, which is automatic for images, and can be easily included in the preprocessing layer. I have however found no similar feature when it comes to action. To be fair, I was never working with normalized observations and it never caused any problem, but for actions it's another story. SAC authors really insist on the fact that their method is robust to non-normalized action, and in practice working with a steering command included in [-0.5, 0.5] and a throttle in [-1, 1] worked perfectly fine. However, less robust methods such as TD3, DDPG or TRPO are really sensitive to that, and I have observed that the action was systematically draged towards saturations with these methods, if there is no normalization. Therefore, make sure to normalize your action in [0.1]
 
 ### Reset randomisation
 
 I have observed that an agent can easily become biased by the ratio of left turns VS rigth turns. It is actually pretty hard to create a reacing track that balances the two, so to prevent that I have implemented a "random reverse" attribute. Everytimes the agent spawns, there 50% chance the world will be reversed, meaning that the lidar that will be fed to agent will be revsersed, and the action he will, chose will be reversed before being fed to AirSim. We have found that it can help a lot to avoid such biases.
 
 ### Actor and critic networks
 
 By default, the actor critic consists of 2 layers of 256 nodes. The input shape is given by whater your feature extractor is ... well extracting 🤔. In the case of a mutli input policy, remember that it will just concatenate a flatten version of every observation (if you are not working with images). The output layer has a size of 2: a command for throttle and a command for steering.
 
![Actor_default](https://user-images.githubusercontent.com/46826148/161277050-785feb99-9e28-4499-976c-01abc81c750c.png)

(I used <a href="https://alexlenail.me/NN-SVG/" target="_blank">this cool website</a> for generating this nice image)
 
 The critic network uses by default the exact same architecture (although the actor and the critic networks are of course independent).

 
 ### Off policy and replay buffer
 SAC is an off policy RL algorithm. It basically means that all the trajectories recorded since the beginning of the learning can be used for every training step. It is tremendously important, because trajectories take a precious time to be recorded in our case, therefore remaking an entire database of trajectories for each policy iteration is not efficient nor feasible, especially for continuous state/action spaces. 
 
 In our case, a replay buffer contains all trajectories informations, namely each observation, state transition and action, along with the matching reward. I want to lay the emphasis on the fact that this guy can quickly become very heavy, especially when working with image observation.
 During each <code>collect_rollout</code>, the trajectories are added to the replay buffer, and during each <code>train</code>, trajectories are randomly sampled from the replay buffer and stochastic gradient descent is done on all networks (we will go further on that point later).
 
 The matter of replay buffer is not as simple as it may sound in the first place, and more advanced replay buffer may lead to significant gain of performances (see Hindsight Experience Replay ). However, I have made the choice of using a regular list, for the sake of simplicity of coding.
 
 ### Pre-training from teacher's demonstration based replay buffer
 What is really nice about the replay buffer is that it is on no account used to evaluate a policy, but rather to learn how to model the environment. My idea on this point is to record the trajectories of a human teacher, to pretrain all the networks. The teacher controls the vehicle with a controller, and drives the car as he normally would, while every action and observations are collected. The network is then trained a first time exclusively on this set of trajectories, which gives (I believe) a good initialisation for the feature extractor networks and the critic network. I am not really sure whether or not pretraining the actor network is a good idea or not, as it seems to overfit to an extent. The learning process is then exactly the same, but starts with a non-empty replay buffer.
 
 *This approach has been strongly influenced by Andrew N. G's "Exploration and Apprenticeship Learning in Reinforcement Learning"*
 
 
 
 
 
 
 
 
 
  
 


