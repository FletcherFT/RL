# Fletcher Thompson's Submission
## Instructions
1. Results and code can be viewed using this repo, however the ROS integration of Gazebo and some special plugins are required to perform training and validation of the policy and value function networks.
2. Run resultsplotter.m in MATLAB to view the training curves of all attempts at the AUV, Pendulum, and Mountain Car Continuous environments.
3. fakegym.py is the interface between the simulator and the network, ROS requires python 2.7 to work properly, so please run any .py files using python2. Browse through the different timestamped folders in AUVSim-v0 to see the different reward functions and stopping conditions trialed.
4. You can modify resultsplotter.m to view different logged variables (you can see all logged options from the 'headers' cell array in MATLAB).
## Further Information
+ The code used in this repo is adapted from [Pat Coady's Github](https://github.com/pat-coady/trpo).
+ Email Fletcher.Thompson@utas.edu.au for further information about the simulator integration.
+ See AUVSim-v0/Dec-30_23-15-41 for a .ipynb implementation of the trainer. 
