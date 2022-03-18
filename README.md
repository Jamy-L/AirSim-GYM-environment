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
      It is then returning an observation following the action, a computed reward estimated for the timestep, a boolean indicating whether or not the episode should be finished, and some debugging infos. In this project, the time step is configurable as a number of frames to simulate or a given simulation time to elapse. It is not really clear which choice makes more sense so... make your choice ?</li>
 <li> <code>reset()</code> basically acts as an initialisation, and is called everytime a simulation ends. That's where all the randomness occurs, to avoid overfitting. It has to return an observation too.</li>
 <li> <code>render()</code> is supposed to be (maybe) called once and to render the simulation each step. Here it's really not that useful since it will run anyway in Unreal Engine, but I have made a little plot of the lidar and camera, although the optimization is terrible (sorry) 😥</il>
 <li> <code> close()</code> is supposed to close the system. In this program it justs gives back the control to the player, so it's a great way to make sure a player can take control. </li>
 <li> <code> spawn_points_checker()</code> Was introduced by me. It just reviews all the spawn points and tests the extreme values of spawn angles for each, so you can visually check if there is no mistake.
</ol>

### Observations and actions
Observation and actions spaces have to be specified while initing the environment in <code> __init__</code>. A lower and upper bound has to be predefined (you can put +infinite so it's not too restrictive), and the types have to be mentionned. It's generally a good idea to simulate a first order memory by adding the previous step's action and observations as an obervation.

I propose to take AirSim's lidar data as an observation vector. The SETTINGS.JSON file contains all the settings related to that (number of points, speed, angle, channel number ...). Keep in mind that the lidar data's size is changing at each step depending on what the lidar is hitting... To that I propose a very simple fix : if there are too many point : crop them. If there are too many : make a padding and just recopy one of the points. Also, <code>jamyys_toolkit.py</code> contains a function to convert the cartasian coordinates to polar. Just make sure to set global coordinates to relative coordinates in SETTINGS.JSON.

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
 <ul>
  "going fast" is implicitly a synonym of marking a lot points, since the episode time is capped at 1000 steps (reached if there is no crash).
 
