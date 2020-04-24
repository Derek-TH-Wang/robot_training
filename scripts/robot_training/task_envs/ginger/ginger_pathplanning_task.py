import os
import sys
import copy
import rospy
import numpy as np
from gym import utils
from gym import spaces
from sensor_msgs.msg import JointState
from robot_training.robot_envs.ginger import ginger_env
from robot_training.util.utility import *


class GingerTaskEnv(ginger_env.GingerEnv, utils.EzPickle):
    def __init__(self):
        rospy.loginfo("go into GingerTaskEnv")
        self.get_params()
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.n_observations, ), dtype='float32')
        super(GingerTaskEnv, self).__init__(use_sim_env=self.use_sim_env)
        rospy.loginfo("========= Out GingerTestEnv")

    def get_params(self):
        # get configuration parameters
        self.use_sim_env = rospy.get_param('/ginger_env/use_sim_env')
        self.n_actions = rospy.get_param('/ginger_env/n_actions')
        self.n_observations = rospy.get_param('/ginger_env/n_observations')
        self.n_dof = rospy.get_param('/ginger_env/n_dof')

        self.start_angle = rospy.get_param('/ginger_env/start_angle')
        self.goal_angle = rospy.get_param('/ginger_env/goal_angle')

        self.action_step = rospy.get_param('/ginger_env/action_step')
        self.step_punishment = rospy.get_param('/ginger_env/step_punishment')
        self.closer_reward_type = rospy.get_param(
            '/ginger_env/closer_reward_type')
        self.step_bonus = rospy.get_param('/ginger_env/step_bonus')
        self.impossible_movement_punishement = rospy.get_param(
            '/ginger_env/impossible_movement_punishement')
        self.reached_goal_reward = rospy.get_param(
            '/ginger_env/reached_goal_reward')

        self.max_distance = rospy.get_param('/ginger_env/max_distance')

        self.init_angle = [self.start_angle["joint0"],
                           self.start_angle["joint1"], self.start_angle["joint2"], self.start_angle["joint3"],
                           self.start_angle["joint4"], self.start_angle["joint5"], self.start_angle["joint6"]]
        self.desired_angle = [self.goal_angle["joint0"],
                              self.goal_angle["joint1"], self.goal_angle["joint2"], self.goal_angle["joint3"],
                              self.goal_angle["joint4"], self.goal_angle["joint5"], self.goal_angle["joint6"]]

        self.init_dist_from_des = 0

    def _set_init_joint(self):
        rospy.logdebug("Init Joint:")
        rospy.logdebug(self.init_angle)

        self.movement_result = self.set_left_arm_position(self.init_angle)
        if self.movement_result:
            # INIT POSE
            rospy.logdebug("Moved To Init Position ")
            self.last_joint_angle = copy.deepcopy(self.init_angle)
            self.last_dist_from_des = self.calculate_distance_between(
                self.desired_angle, self.last_joint_angle)
            self.init_dist_from_des = self.last_dist_from_des
            rospy.logdebug("INIT DISTANCE FROM GOAL==>" +
                           str(self.last_dist_from_des))
        else:
            rospy.logfatal("Moved To Init Position ERR")
            assert False, "Desired GOAL EE is not possible"

        return self.movement_result

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        The simulation will be paused, therefore all the data retrieved has to be
        from a system that doesnt need the simulation running, like variables where the
        callbackas have stored last know sesnor data.
        :return:
        """
        rospy.logdebug("Init Env Variables...")
        rospy.logdebug("Init Env Variables...END")

    def _set_action(self, action):
        """
        every joint has 3 choce: -action_step, 0, +action_step
        n_dof robot has 3**n_dof action_space
        """
        rospy.logdebug("action = " + str(action))
        current_joint_angle = [0]*self.n_dof
        temp_action = action
        for i in range(self.n_dof):
            temp_compare = (3**self.n_dof)/(3**(i+1))
            if temp_action < temp_compare:
                current_joint_angle[i] = self.last_joint_angle[i] - \
                    self.action_step
                temp_action = temp_action - 0*temp_compare
            elif temp_action >= temp_compare and temp_action < temp_compare*2:
                current_joint_angle[i] = self.last_joint_angle[i]
                temp_action = temp_action - 1*temp_compare
            else:
                current_joint_angle[i] = self.last_joint_angle[i] + \
                    self.action_step
                temp_action = temp_action - 2*temp_compare

        # Apply action to simulation.
        rospy.logdebug("set = " + str(current_joint_angle))
        self.movement_result = self.set_left_arm_position(current_joint_angle)
        if self.movement_result:
            self.last_joint_angle = copy.deepcopy(current_joint_angle)
        else:
            rospy.logdebug("Impossible joint Position...." +
                           str(current_joint_angle))

    def _get_obs(self):
        current_joint_angle = self.get_left_arm_position()
        rospy.logdebug("get = " + str(current_joint_angle))
        obs = copy.deepcopy(current_joint_angle)
        for i in range(self.n_dof):
            obs.append(self.init_angle[i])
        for i in range(self.n_dof):
            obs.append(self.desired_angle[i])
        for i in range(self.n_dof):
            obs.append(self.desired_angle[i] - current_joint_angle[i])
        norm_dist_from_des = self.calculate_distance_between(
            self.desired_angle, current_joint_angle)
        obs.append(norm_dist_from_des)
        rospy.logdebug("OBSERVATIONS====>>>>>>>"+str(obs))
        return obs

    def _is_done(self, observations):
        """
        If the latest Action didnt succeed, it means that tha position asked was imposible therefore the episode must end.
        It will also end if it reaches its goal.
        """
        current_joint = observations[:self.n_dof]
        done, info = self.calculate_if_done(
            self.movement_result, self.desired_angle, current_joint)
        return done, info

    def _compute_reward(self, observations, done):
        """
        We punish each step that it passes without achieveing the goal.
        Punishes differently if it reached a position that is imposible to move to.
        Rewards getting to a position close to the goal.
        """
        current_pos = observations[:self.n_dof]
        norm_dist_from_des = observations[-1]

        reward = self.calculate_reward(
            self.movement_result, self.desired_angle, current_pos, norm_dist_from_des)
        rospy.logdebug(">>>REWARD>>>"+str(reward))

        return reward

    def calculate_if_done(self, movement_result, desired_angle, current_pos):
        """
        It calculated whather it has finished or not
        """
        done = False
        info = {'reach_goal': False}

        if movement_result:
            position_similar = np.all(np.isclose(
                desired_angle, current_pos, atol=1e-02))
            if position_similar:
                done = True
                info = {'reach_goal': True}
                rospy.logfatal("Reached a Desired Position!")
        else:
            done = True
            # rospy.logfatal("movement_result is wrong")

        return done, info

    def calculate_reward(self, movement_result, desired_angle, current_pos, norm_dist_from_des):
        """
        It calculated whather it has finished or nota and how much reward to give
        """

        if movement_result:
            position_similar = np.all(np.isclose(
                desired_angle, current_pos, atol=1e-02))

            # Calculating Distance
            rospy.logdebug("desired_angle="+str(desired_angle))
            rospy.logdebug("current_pos="+str(current_pos))
            rospy.logdebug("self.last_dist_from_des=" +
                           str(self.last_dist_from_des))
            rospy.logdebug("norm_dist_from_des=" + str(norm_dist_from_des))

            reward = 0.0
            if position_similar:
                reward += self.reached_goal_reward
                rospy.logfatal(
                    "Reached a Desired Position! reward = "+str(reward))
            else:
                if self.closer_reward_type == 0:  # will append different reward calculation method in the future
                    if norm_dist_from_des - self.last_dist_from_des >= 0.0:
                        reward = self.step_punishment
                    else:
                        reward = self.step_bonus
                elif self.closer_reward_type == 1:
                    #reward = 1/(norm_dist_from_des+(1/self.reached_goal_reward)) - 1/(self.init_dist_from_des + (1/self.reached_goal_reward))
                    reward = -(100/pow((self.init_dist_from_des), 3)
                               )*pow((norm_dist_from_des - self.init_dist_from_des), 3)
                #reward += self.step_punishment

            rospy.logdebug("norm_dist_from_des = " + str(round(norm_dist_from_des, 2)
                                                         ) + ", reward = " + str(round(reward, 2)))
        else:
            reward = self.impossible_movement_punishement
            rospy.logdebug("movement_result is wrong")

        # We update the distance
        self.last_dist_from_des = norm_dist_from_des
        rospy.logdebug("Updated Distance from GOAL==" +
                       str(self.last_dist_from_des))
        return reward

    def calculate_distance_between(self, v1, v2):
        """
        Calculated the Euclidian distance between two vectors given as python lists.
        """
        dist = np.linalg.norm(np.array(v1)-np.array(v2))
        return dist
