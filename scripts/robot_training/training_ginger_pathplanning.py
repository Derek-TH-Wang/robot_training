#!/usr/bin/env python3
import os
import sys
import threading
import rospy
import rospkg
import gym
from robot_training.util.utility import EnvRegister
from robot_training.rl_algorithms.DQNRobotSolver import DQNRobotSolver


def thread_job():
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node('ginger_training_pathplanning',
                    anonymous=True, log_level=rospy.INFO)

    rospy.loginfo("read parm from yaml")
    task_env_name = rospy.get_param('/ginger_training/task_and_robot_environment_name')
    rospy.loginfo("task_and_robot_environment_name ==>" + str(task_env_name))
    alpha = rospy.get_param('/ginger_training/alpha')
    alpha_decay = rospy.get_param('/ginger_training/alpha_decay')
    gamma = rospy.get_param('/ginger_training/gamma')
    epsilon = rospy.get_param('/ginger_training/epsilon')
    epsilon_log_decay = rospy.get_param('/ginger_training/epsilon_decay')
    epsilon_min = rospy.get_param('/ginger_training/epsilon_min')
    replay_buffer_size = rospy.get_param('/ginger_training/replay_buffer_size')
    batch_size = rospy.get_param('/ginger_training/batch_size')
    n_episodes_training = rospy.get_param('/ginger_training/episodes_training')
    n_episodes_running = rospy.get_param('/ginger_training/episodes_running')
    n_win_ticks = rospy.get_param('/ginger_training/n_win_ticks')
    min_episodes = rospy.get_param('/ginger_training/min_episodes')
    monitor = rospy.get_param('/ginger_training/monitor')
    quiet = rospy.get_param('/ginger_training/quiet')
    n_actions = rospy.get_param('/ginger_env/n_actions')
    n_observations = rospy.get_param('/ginger_env/n_observations')
    action_step = rospy.get_param('/ginger_env/action_step')
    reached_goal_reward = rospy.get_param('/ginger_env/reached_goal_reward')

    max_env_steps = None

    rospackage_name = "robot_training"
    model_name = "ginger_dqn"

    # add thread, deal with ros callback function
    add_thread = threading.Thread(target=thread_job)
    add_thread.start()
    rospy.sleep(1)

    rospy.loginfo("env_register")
    task_env = EnvRegister(task_env_name)
    env_object = gym.make(task_env)

    rospy.loginfo("Starting Learning")
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

    rospy.logfatal("training over")
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    outdir = pkg_path + '/learning_result/models'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        rospy.loginfo("Created folder="+str(outdir))

    agent.save(model_name, outdir)
    agent.load(model_name, outdir)

    agent.run(num_episodes=n_episodes_running, do_train=False)

    rospy.spin()
