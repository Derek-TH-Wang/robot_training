#!/usr/bin/env python
import os
import sys
import rospy
import rospkg
from robot_training.util.utility import EnvRegister
from robot_training.rl_algorithms.DQNRobotSolver import DQNRobotSolver

if __name__ == '__main__':
    rospy.init_node('fetch_n1try_algorithm',
                    anonymous=True, log_level=rospy.WARN)

    rospy.logwarn("read parm from yaml")
    task_env_name = rospy.get_param('/fetch/task_and_robot_environment_name')
    rospy.logwarn("task_and_robot_environment_name ==>" + str(task_env_name))
    n_observations = rospy.get_param('/fetch/n_observations')
    n_actions = rospy.get_param('/fetch/n_actions')
    n_episodes_training = rospy.get_param('/fetch/episodes_training')
    n_episodes_running = rospy.get_param('/fetch/episodes_running')
    n_win_ticks = rospy.get_param('/fetch/n_win_ticks')
    min_episodes = rospy.get_param('/fetch/min_episodes')
    max_env_steps = None
    gamma = rospy.get_param('/fetch/gamma')
    epsilon = rospy.get_param('/fetch/epsilon')
    epsilon_min = rospy.get_param('/fetch/epsilon_min')
    epsilon_log_decay = rospy.get_param('/fetch/epsilon_decay')
    alpha = rospy.get_param('/fetch/alpha')
    alpha_decay = rospy.get_param('/fetch/alpha_decay')
    batch_size = rospy.get_param('/fetch/batch_size')
    replay_buffer_size = rospy.get_param('/ginger/replay_buffer_size')
    monitor = rospy.get_param('/fetch/monitor')
    quiet = rospy.get_param('/fetch/quiet')
    reached_goal_reward = rospy.get_param('/fetch/reached_goal_reward')
    rospackage_name = "robot_training"
    model_name = "fetch_dqn"

    rospy.logwarn("env_register")
    env_object = EnvRegister(task_env_name)

    rospy.logwarn("Starting Learning")
    agent = DQNRobotSolver(env_object,
                           task_env_name,
                           n_observations,
                           n_actions,
                           n_win_ticks,
                           min_episodes,
                           max_env_steps,
                           gamma,
                           epsilon,
                           epsilon_min,
                           epsilon_log_decay,
                           reached_goal_reward,
                           alpha,
                           alpha_decay,
                           batch_size,
                           replay_buffer_size,
                           monitor,
                           quiet)
    agent.run(num_episodes=n_episodes_training, do_train=True)

    rospy.logwarn("training over")
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    outdir = pkg_path + '/models'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        rospy.logfatal("Created folder="+str(outdir))

    agent.save(model_name, outdir)
    agent.load(model_name, outdir)

    agent.run(num_episodes=n_episodes_running, do_train=False)
