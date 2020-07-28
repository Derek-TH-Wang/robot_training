# robot_training  
Robot reinforcement lerning training demo, using OpenAI Gym environment with Ros  

## system equirement:  
ubuntu18.04, ros-melodic, pytorch1.4, tensorflow1.13+, cuda10+  
```
sudo pip3 install rospkg
sudo pip3 install defusedxml
sudo pip3 install tqdm
sudo pip2 install keras
sudo pip3 install keras
sudo apt-get install ros-melodic-costmap-2d
sudo apt-get install ros-melodic-robot-controllers
sudo apt-get install ros-melodic-effort-controllers
```

## start training:  
### fetch robot dqn training demo:  
```
git clone https://bitbucket.org/theconstructcore/fetch_tc.git
git checkout melodic-gazebo9
git clone https://bitbucket.org/theconstructcore/spawn_robot_tools.git
catkin_make
source devel/setup.bash
roslaunch fetch_moveit_config demo.launch
roslaunch robot_training fetch_training.launch
```
### iriwam robot q-learning demo:  
```
sudo apt-get install ros-melodic-joint-trajectory*
git clone https://bitbucket.org/theconstructcore/iri_wam.git
git checkout kinetic-gazebo9
(cp start_world.launch and put_robot_in_world.launch in master branch to kinetic-gazebo9 branch)
git clone https://bitbucket.org/theconstructcore/hokuyo_model.git
catkin_make
source devel/setup.bash
roslaunch robot_training iriwam_training.launch
```
### ginger robot dqn training demo:  
```
git clone https://github.com/Derek-TH-Wang/gingerurdf.git
git checkout simulator
catkin_make
source devel/setup.bash
```
simulator switch:  
set 'use_sim_env' parameter True or False in config/ginger_env_parm.yaml  

if use tianshou framework(multi env parallel training cannot use simulator yet):  
```
roslaunch robot_training ginger_training_pathplanning_ts.launch
```
if not use tianshou framework:  
```
roslaunch gingerurdf simulator.launch
roslaunch robot_training ginger_training_pathplanning.launch
```
