# What file is making what ? #
<ul>
  <li>Airsim_gym_env.py is self explanatory. It's where the magic happens between AirSim and GYM</li>
  <li>Reinforcement learning.py calls Stable Baselines and uses the gym env to train an agent</li>
  <li>jamys_toolkit.py contains all useful functions not directly related to the gym env. It concerns lidar formatting, action normalisation, custom feature extractor and the implementation of checkpoints</li>
  <li>sim_to_real_library.py contains functions that we have used to transpose the agent to a real environment, namely changing lidar angles and applying a scaling factor to radius value</li>
  <li>Onboard_implementation.py is the algorithm that was implemented on the real car. It uses multithreading to sample observations and take a decision, with a state machine (using the neural network trained on AirSim). </li>
</ul>
