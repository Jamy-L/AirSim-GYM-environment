# AirSim GYM environment
This set of programs aims to make reinforcement learning on AirSim easier, by providing a GYM custom environment. It is fully compatible with Stable Baselines 3 which I recommend using.

## Beginning
Make sure to be familiar with AirSim (https://microsoft.github.io/AirSim/). It requires Unreal Engine 4 and some python modules like open CV. Cuda is also strongly recommended, and you may want to use a beefy GPU with at least 16Go of RAM to be serein.

You should also check GYM (https://gym.openai.com/) which is a python environment library that basically makes reinforcement learning much easier to program.

Stable Baselines 3 provides a set of very performent and modulable RL algorithms and is based on PyTorch. Make sure to have a clean install of the latter (https://stable-baselines3.readthedocs.io/en/master/)

## Presentation
A GYM environment is basically composed of 3 methods : 
<ol>
 <li> <code>step()</code> takes an action input and performs it on the system.
      It is then returning an observation following the action, a computed reward estimated for the timestep, a boolean indicating whether or not the episode should be finished, and some debugging infos. In this project, the time step is configurable as a given time to elapse before further decision making. Please read the dedicated section for more information on this choice</li>
 <li> <code>reset()</code> basically acts as an initialisation, and is called everytime a simulation ends. That's where all the randomness occurs, to avoid overfitting. It has to return an observation too.</li>
 <li> <code>render()</code> is supposed to be (maybe) called once and to render the simulation each step. Here it's really not that useful since it will run anyway in Unreal Engine, but I have made a little plot of the lidar and camera, although the optimization is terrible (sorry) 😥</il>
 <li> <code> close()</code> is supposed to close the system. In this program it justs gives back the control to the player, so it's a great way to make sure a player can take control. </li>
 <li> <code> spawn_points_checker()</code> Was introduced by me. It just reviews all the spawn points and tests the extreme values of spawn angles for each, so you can visually check if there are no mistake.
</ol>

### Observations and actions
Observation and actions spaces have to be specified while initiating the environment in <code> __init__</code>. A lower and upper bound have to be predefined (you can put +infinite so it's not too restrictive), and the types have to be mentionned. It's generally a good idea to simulate a first order memory by adding the previous step's action and observations as an obervation.

I propose to take AirSim's lidar data as an observation vector. The SETTINGS.JSON file contains all the settings related to that (number of points, speed, angle, channel number ...). Keep in mind that the lidar data's size is changing at each step depending on what the lidar is hitting... To that I propose a very simple fix : if there are too many point : crop them. If there are too little points : make a padding and just recopy one of the points. Also, <code>jamyys_toolkit.py</code> contains a function to convert the cartasian coordinates to polar. Just make sure to set global coordinates to relative coordinates in SETTINGS.JSON.

Camera feed is also a good observation source, but it's very expensive.




Actions basically consists of the throttle and the steering, nothing more.

## Reward function
### Checkpoints and spawns
I have implemented a random respawn and checkpoint system. Just add collision-free objects in Unreal Engine (like transparent Spheres) to mark the checkpoints, and add their coordinates in the programm. You also need to give the coordinates of the "player spawn" object in the UE project because the coordinates given by the AirSim API are relative to it. Make sure to give them in the right order !. The <code> Circuit_wrapper</code> class of <code> jamys_toolkit.py</code> will directly make all the conversion, and make a circular infinite loop.

For respawn points, you can make the same thing and specify a minimum and maximal deviation angle (when spawning, a random value will be sampled between the two). You also need to manually specify what is the index of the first checkpoint to validate, since it depends on where the car is spawning.

 ### Reward
 Reward is calculated as : 
 <ul>
 <li> -100 points in case of crash, and the episode is ending </li>
 <li> +50 points when a checkpoint is validated </li>
 <li> -0.1*speed points when the speed is below is under 1 m/s (no slowpoke on the racing track !) </li>
 </ul>
  "going fast" is implicitly a synonym of marking a lot points, since the episode time is capped at 1000 steps (reached if there is no crash).
  
  ## About time steps and simulation speed
  This is a very though matter. Simulating a Markov decision process generally requires to chose a time step separing two action. Typically, GYM scheme is "fetch observation"-"choose action"-"calling a simulation step". To simulate this scheme, my method is to have the simulation paused at all time, except for the specific window where <code> step()</code> is called. In itself, this method has it flaws because real environment will not pause when calculations are made. However, the delays involved by inference are neglectable with regards to the delays involed with training a network. I believe the key element is to ensure that observations (both for a simulated and real environment) are fetched with regular time steps, because it enables some sort of speed estimation based on a first order memory of spacial observation. This behaviour is achievable on a real system thanks to onboard OS and proper task ordonnancing.
  
### What time is it?
  However, that being said, the time step problem is far from being solved. When speaking of clocks/frequencies/framerate/speed, many protagonists come to action.
  <ul>
 <li> Unreal Engine is a video games engine that need to generate a certain number of frames per second. The FPS are dependant on the hardware running the environment, and on the system's different active tasks. The FPS are not only a visual asset, they also intervene in the physics engine, and must be kept around a reasonable value. Typically, staying over 30 FPS is a must, and I believe it's impossible to get over 120 FPS. </li>
 <li> PhysX is taking care of the vehicule's physics, by applying a finite difference scheme over time. This time step in particular is not easily reachable, but is tremendously important for simulating a realistic behaviour. For that reason, we have to ensure that this time step is not too big, because it would significantly worsen realism. For example, collisions would not be properly detected. I have also experimentally observed that when this time step gets too big, the dynamic behaviour changes a lot. A constant acceleration on a straight line would not get the car as far as it would with a shorter time step, although they are driving for the same duration with the same speed. I will delve further on this issue later </li>
 <li> AirSim also has it's inner clock. It can be scaled in SETTINGS.JSON, which is really handy to speed up trainign by speeding simulations. </li>
 <li> The computer running the simulation itself has its own real time clock. <code> time.sleep()</code> enables to wait for a given real time between samples</li>
 </ul>
 
### So which clock should be used?
 A first idea is to use AirSim API's <code>simContinueForFrames()</code> or <code>simContinueForTime()</code>. However, I have found that this approach is not that good. First of all, a "frame step" doesn't really makes sense in my opinion because it's very hard to transpose on a real environment. Furthermore, framerate drops due to external activities must be taken into account, and it impossible to predict how the framerate will react when you are doubling the simulation clock.
 
 <code>simContinueForTime()</code> Sounds good, but is a trap in my opinion. It is very inconsistent with short time steps, and clock scaling factors. This feature is discussed among the AirSim community, so I have decided to completely abandon it.
 
 My conclusion is that <code> time.sleep()</code> is the stepping method to use (you can of course use a dedicated thread instead, if you find it useful). Let's say for now that the time sclaing factor is 1. My main argument is that Unreal Engine is meant to simulate real time environments, with framerates above 30 FPS. Therefore, in that scenario, 1s in real life / on the processor clock accounts for 1s in the simulation. The FPS are dynamically adjusting to ensure that the simulation is indeed real time, although this objective cannot be achived if the FPS are dropping below 30 FPS (in that case the simulation clock will become slower than the real world clock). When the scaling factor is different than 1 (let's say it is bigger), physX needs to compute more time steps during a given time. We may talk of "effective framerate" equal to apparent framerate divided by clock scaling factor (FPS/Scale). For example, if my computer runs AirSim at 40 FPS, and I set a scaling factor of 4, it means that the effective FPS will be 10, which is much below 30FPS. In that case, PhysX time step will not be precise enough, and the physics will not be precise at all.
 
 Therefore, I believe that the tradoff is to find the biggest scaling factor verifying FPS/factor >= 30. Since UE doesn't seem to go above 120FPS, a scaling factor bewteen 3 and 4 seems optimal (if your machine can run it at 120 FPS of course). The stepping method should thus call <code> time.sleep(time/scale)</code> or something similar.
 
 # My implementation of reinforcment learning methods to autonomous driving
 
 The AirSim enviornment uses a continuous action space and a continuous observtion space. For clarity, the observations are of type "Multi Input" since lidar data, previous actions, and a picture are fetched every step. The action space is just a 2D vector containing steering and thorttle, therefore it is no multi input.
 
 The continuity of the problems narrows the choice of RL algorithms. Furthermore, due to the nature of the environment, observation data will be very scarce, which gives off-policy algorithms a lead against on-policy algorithms.
 
 These details being cleared, SAC and TD3 appear to be the most logical choice. I have personally chosen to use SAC because I was more familliar with it.
 
 ## Workings with SAC
 SAC is explained in more details here : https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
 It is strongly recommended to read the original article to understand the inner core of the algorithm.
 
 In a pragmatic way, three protagonists come to action during the training : 
 <ul>
 <li> A critic network learns to predict the expected reward for any action, in a given state. There are actually 2 critic networks being trained at each step, and the best one is selected every time.</li>
 <li> The entropy coefficient bascially regulates whether policies will rather explore or exploit the environment. It is evolving during the training, and helps a long to stabilize the learning process. Generally the coeffcient starts at a high value, and converges toward 0 at the end of training.</li>
 
 <li> The actor network maps a probability to every action for each given state </li>
 </ul>
 
 All these protagonists evolve at the same time, their training is tightly linked. The trainings are of course based on collected trajectories, on which the training is done. After training, a policy is deployed on the environment to collect more trajectories (it may be trajectoiries of exploration or exploitation), and the process cycles. These two process are called <code>train</code> and <code>collect_rollout</code>
 
 ### Off policy and replay buffer
 SAC is an off policy RL algorithm. It basically means that all trajectories recorded since the beginning of the learning can be used for every training step. It is tremendously important, because trajectories take a precious time to be recorded in our case, therefore remaking an entire database of trajectories for each policy iteration is not efficient nor feasable ... 
 
 In our case, a replay buffer contains all trajectories informations, namely each observation, state transition and action, along with the matching reward. I want to lay the emphasis on the fact that this guy can quickly become very heavy, especially when working with image observation.
 During each <code>collect_rollout</code>, the trajectories are added to the replay buffer, and during each <code>train</code>, trajectories are randomly sampled from the replay buffer and stochastic gradient descent is done on all networks (we will go further on that point later).
 
 The matter of replay buffer is not as simple as it may sound in the first place, and more advanced replay buffer may lead to significant gain of performances (see Hindsight Experience Replay ). However, I have made the choice of using a regular list, for the sake of simplicty of coding.
 
 ### Pre-training from teacher's demontration based replay buffer
 What is really nice about the replay buffer is that it is on no account used to evaluate a policy, but rather to learn how to model the environment. My idea on this point is to record the trajectories of a human teacher, to pretrain all the networks. The teacher controls the vehicle with a controller, and drives the car as he nromally would, while every action and observations are collected. The network is then trained a first time exclusively in this set of trajectories, which gives (I believe) a good initialisation for the feature extractor networks and the critic network. I am not really sure whether or not pretraining the actor newtwork is a good idea or not, as it seems to overfitt to an extent. The learning process is then exactly the same, but starts with a non empty replay buffer.
 
 This approach has been strongly influenced by Andrew N.G's "Exploration and Apprenticeship Learning in Reinforcement Learning"
 
 
 
 
 
 
 
 
  
 
