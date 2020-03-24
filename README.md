# robot_training  
Robot reinforcement lerning training using OpenAI Gym in Ros Env  

## system equirement:  
ubuntu18.04, ros-melodic, python2 tensorflow1.13.0, cuda10  

## start training:  
### fetch robot demo:  
```
git clone https://bitbucket.org/theconstructcore/fetch_tc.git  
git checkout melodic-gazebo9  
git clone https://bitbucket.org/theconstructcore/spawn_robot_tools.git  
catkin_make  
source devel/setup.bash  
roslaunch fetch_moveit_config demo.launch  
roslaunch robot_training fetch_training.launch  
```
### iriwam robot demo:  
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

