import gym
from gym import envs
from gym.envs.registration import register
import rospy
import rosparam
import rospkg
import os
import roslaunch
import sys
import subprocess


def EnvRegister(task_env, max_episode_steps=10000):
    if task_env == 'FetchTest-v0':
        rospy.loginfo("register FetchTest-v0")
        register(
            id=task_env,
            entry_point='task_envs.fetch.fetch_test_task:FetchTestEnv',
            max_episode_steps=max_episode_steps,
        )
        from task_envs.fetch import fetch_test_task
    elif task_env == 'IriWamTcpToBowl-v0':
        rospy.loginfo("register IriWamTcpToBowl-v0")
        register(
            id=task_env,
            entry_point='task_envs.iriwam.iriwam_test_task:IriWamTcpToBowlEnv',
            max_episode_steps=max_episode_steps,
        )
        from task_envs.iriwam import iriwam_test_task
    elif task_env == 'GingerPathPlanning-v0':
        rospy.loginfo("GingerPathPlanning-v0")
        register(
            id=task_env,
            entry_point='task_envs.ginger.ginger_pathplanning_task:GingerTaskEnv',
            max_episode_steps=max_episode_steps,
        )
        from task_envs.ginger import ginger_pathplanning_task
    else:
        rospy.loginfo("register None")
        return None
    # We check that it was really registered
    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    assert (task_env in env_ids), "The Task_Robot_ENV given is not Registered ==>" + str(task_env)

    rospy.loginfo(
        "Register of Task Env went OK, lets make the env..."+str(task_env))
    return task_env
    # env = gym.make(task_env)
    # rospy.logwarn("gym make finish")
    # return env


def LoadYamlFileParamsTest(rospackage_name, rel_path_from_package_to_file, yaml_file_name):

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file)
    path_config_file = os.path.join(config_dir, yaml_file_name)

    paramlist = rosparam.load_file(path_config_file)

    for params, ns in paramlist:
        rosparam.upload_params(ns, params)


class ROSLauncher(object):
    def __init__(self, rospackage_name, launch_file_name, ros_ws_abspath="/home/derek/openai_ws"):

        self._rospackage_name = rospackage_name
        self._launch_file_name = launch_file_name

        self.rospack = rospkg.RosPack()
        pkg_path = self.rospack.get_path(rospackage_name)
        if pkg_path:
            # If the package was found then we launch
            rospy.loginfo(
                ">>>>>>>>>>Package found in workspace-->"+str(pkg_path))
            launch_dir = os.path.join(pkg_path, "launch")
            path_launch_file_name = os.path.join(launch_dir, launch_file_name)

            rospy.logwarn("path_launch_file_name=="+str(path_launch_file_name))

            source_env_command = "source "+ros_ws_abspath+"/devel/setup.bash;"
            roslaunch_command = "roslaunch  {0} {1}".format(
                rospackage_name, launch_file_name)
            command = source_env_command+roslaunch_command
            rospy.logwarn("Launching command="+str(command))

            p = subprocess.Popen(command, shell=True)

            state = p.poll()
            if state is None:
                rospy.loginfo("process is running fine")
            elif state < 0:
                rospy.loginfo("Process terminated with error")
            elif state > 0:
                rospy.loginfo("Process terminated without error")
            """
            self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(self.uuid)
            self.launch = roslaunch.parent.ROSLaunchParent(
                self.uuid, [path_launch_file_name])
            self.launch.start()
            """
            rospy.loginfo(">>>>>>>>>STARTED Roslaunch-->" +
                          str(self._launch_file_name))
        else:
            rospy.logfatal("No package found")
