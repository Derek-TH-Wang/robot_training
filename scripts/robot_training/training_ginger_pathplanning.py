#!/usr/bin/env python
import os
import sys
import threading
import rospy
import rospkg
from robot_training.util.utility import EnvRegister
from robot_training.rl_algorithms.DQNRobotSolver import DQNRobotSolver


def thread_job():
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node('ginger_training_pathplanning',
                    anonymous=True, log_level=rospy.INFO)

    rospy.loginfo("read parm from yaml")
    task_env_name = rospy.get_param('/ginger/task_and_robot_environment_name')
    rospy.loginfo("task_and_robot_environment_name ==>" + str(task_env_name))
    alpha = rospy.get_param('/ginger/alpha')
    alpha_decay = rospy.get_param('/ginger/alpha_decay')
    gamma = rospy.get_param('/ginger/gamma')
    epsilon = rospy.get_param('/ginger/epsilon')
    epsilon_log_decay = rospy.get_param('/ginger/epsilon_decay')
    epsilon_min = rospy.get_param('/ginger/epsilon_min')
    batch_size = rospy.get_param('/ginger/batch_size')
    n_episodes_training = rospy.get_param('/ginger/episodes_training')
    n_episodes_running = rospy.get_param('/ginger/episodes_running')
    n_win_ticks = rospy.get_param('/ginger/n_win_ticks')
    min_episodes = rospy.get_param('/ginger/min_episodes')
    monitor = rospy.get_param('/ginger/monitor')
    quiet = rospy.get_param('/ginger/quiet')
    n_actions = rospy.get_param('/ginger/n_actions')
    n_observations = rospy.get_param('/ginger/n_observations')
    # action_upper_limit = rospy.get_param('/ginger/action_upper_limit')
    # action_lower_limit = rospy.get_param('/ginger/action_lower_limit')
    action_step = rospy.get_param('/ginger/action_step')

    max_env_steps = None

    rospackage_name = "robot_training"
    model_name = "ginger_dqn"

    # add thread, deal with ros callback function
    add_thread = threading.Thread(target=thread_job)
    add_thread.start()
    rospy.sleep(1)

    rospy.loginfo("env_register")
    env_object = EnvRegister(task_env_name)

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
                           alpha,
                           alpha_decay,
                           batch_size,
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
