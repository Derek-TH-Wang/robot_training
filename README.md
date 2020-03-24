# robot_training  
Robot reinforcement lerning training using OpenAI Gym in Ros Env  

## system equirement:  
ubuntu18.04, ros-melodic, python2 tensorflow1.13.0, cuda10  

## prepare work:  
```
git clone https://bitbucket.org/theconstructcore/fetch_tc.git  
git checkout melodic-gazebo9  

git clone https://bitbucket.org/theconstructcore/spawn_robot_tools.git  

catkin_make  
```
## start training:  
fetch robot demo:  
```
source devel/setup.bash  
roslaunch fetch_moveit_config demo.launch  
roslaunch robot_training fetch_training.launch  
```
