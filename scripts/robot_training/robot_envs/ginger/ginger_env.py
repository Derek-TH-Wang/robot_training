import sys
import copy
import rospy
import numpy as np
import geometry_msgs.msg
import trajectory_msgs.msg
from sensor_msgs.msg import JointState
from gingerurdf.msg import ArmMsgs
from gingerurdf.msg import BodyMsgs
from gingerurdf.msg import HandMsgs
from gingerurdf.msg import HeadMsgs
from robot_training.util.utility import *
from robot_training.robot_sim.rviz_envs import robot_rviz_env


class GingerEnv(robot_rviz_env.RobotRvizEnv):

    def __init__(self, use_sim_env = False):
        rospy.loginfo("go into GingerEnv")

        self.use_sim_env = use_sim_env

        self.controllers_list = []

        self.robot_name_space = ""
        self.reset_controls = False

        super(GingerEnv, self).__init__()

        # We Start all the ROS related Subscribers and publishers
        # self.JOINT_STATES_SUBSCRIBER = '/ginger/joint_states'
        self.wheel_name = ["Wheel_left", "Wheel_right", "Wheel_back"]
        self.main_body_name = ["Knee", "Back_Z", "Back_X", "Back_Y"]
        self.head_body_name = ["Neck_Z", "Neck_X", "Head"]
        self.left_arm_name = ["Left_Shoulder_X", "Left_Shoulder_Y",
                              "Left_Elbow_Z", "Left_Elbow_X",
                              "Left_Wrist_Z", "Left_Wrist_X", "Left_Wrist_Y"]
        self.right_arm_name = ["Right_Shoulder_X", "Right_Shoulder_Y",
                               "Right_Elbow_Z", "Right_Elbow_X",
                               "Right_Wrist_Z", "Right_Wrist_X", "Right_Wrist_Y"]
        self.left_hand_name = ["Left_1_1", "Left_2_1", "Left_3_1", "Left_4_1", "Left_5_1",
                               "Left_1_2", "Left_2_2", "Left_3_2", "Left_4_2", "Left_5_2",
                               "Left_1_3", "Left_2_3", "Left_3_3", "Left_4_3", "Left_5_3"]
        self.right_hand_name = ["Right_1_1", "Right_2_1", "Right_3_1", "Right_4_1", "Right_5_1",
                                "Right_1_2", "Right_2_2", "Right_3_2", "Right_4_2", "Right_5_2",
                                "Right_1_3", "Right_2_3", "Right_3_3", "Right_4_3", "Right_5_3"]
        self.joint_state_name = self.main_body_name + self.head_body_name + self.left_arm_name + self.right_arm_name + \
            self.left_hand_name + self.right_hand_name
        self.main_body_lower_limit = [-0.627794, -
                                      1.570796, -0.781907, -0.720820]
        self.main_body_upper_limit = [0.539830, 1.570796, 0.991347, 0.720820]
        # have to increase joint12,19,14,21 range
        self.left_arm_lower_limit = [-2.26, -0.16231, -
                                     1.570796, -1.570796, -1.570796, -0.738274, -0.523598]
        self.left_arm_upper_limit = [
            0.7854, 1.730843, 0.785398, 0.3, 1.570796, 0.738274, 0.523598]
        self.right_arm_lower_limit = [-2.26, -1.730843, -
                                      0.785398, -1.570796, -1.570796, -0.738274, -0.523598]
        self.right_arm_upper_limit = [
            0.7854, 0.16231, 1.570796, 0.3, 1.570796, 0.738274, 0.523598]

        self._check_all_systems_ready()

        # self.js_sub = rospy.Subscriber(
        #     self.JOINT_STATES_SUBSCRIBER, JointState, self.sub_js_callback)
        # self.get_joint_states = JointState()

        # self.js_pub = rospy.Publisher(
        #     '/joint_states', JointState, queue_size=5)
        # self.set_joint_states = JointState()

        self.main_body_joint_pub = rospy.Publisher(
            "/MainBody/TargetPosition", BodyMsgs, queue_size=5)
        self.head_body_joint_pub = rospy.Publisher(
            "/HeadBody/TargetPosition", HeadMsgs, queue_size=5)
        self.main_body_joint_pub = rospy.Publisher(
            "/MainBody/TargetPosition", BodyMsgs, queue_size=5)
        self.head_body_joint_pub = rospy.Publisher(
            "/HeadBody/TargetPosition", HeadMsgs, queue_size=5)
        self.left_arm_joint_pub = rospy.Publisher(
            "/LeftArm/TargetPosition", ArmMsgs, queue_size=5)
        self.right_arm_joint_pub = rospy.Publisher(
            "/RightArm/TargetPosition", ArmMsgs, queue_size=5)
        self.left_hand_joint_pub = rospy.Publisher(
            "/LeftHand/TargetPosition", HandMsgs, queue_size=5)
        self.right_hand_joint_pub = rospy.Publisher(
            "/RightHand/TargetPosition", HandMsgs, queue_size=5)

        self.main_body_joint = None
        self.head_body_joint = None
        self.left_arm_joint = None
        self.right_arm_joint = None
        self.left_hand_joint = None
        self.right_hand_joint = None
        rospy.Subscriber("/MainBody/Position", BodyMsgs,
                         self.sub_main_body_joint_callback)
        rospy.Subscriber("/HeadBody/Position", HeadMsgs,
                         self.sub_head_body_joint_callback)
        rospy.Subscriber("/LeftArm/Position", ArmMsgs,
                         self.sub_left_arm_joint_callback)
        rospy.Subscriber("/RightArm/Position", ArmMsgs,
                         self.sub_right_arm_joint_callback)
        rospy.Subscriber("/LeftHand/Position", HandMsgs,
                         self.sub_left_hand_joint_callback)
        rospy.Subscriber("/RightHand/Position",
                         HandMsgs, self.sub_right_hand_joint_callback)

        
        # parallel calculation, no need of rviz env
        self.get_main_body_joint = [0.0]*4
        self.get_head_body_joint = [0.0]*3
        self.get_left_arm_joint = [0.0]*7
        self.get_right_arm_joint = [0.0]*7
        self.get_left_hand_joint = [0.0]*5
        self.get_right_hand_joint = [0.0]*5

        rospy.loginfo("========= Out GingerRobotEnv")

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        if self.use_sim_env:
            self._check_all_sensors_ready()
        return True

    def _check_all_sensors_ready(self):
        self._check_rostopic_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_rostopic_ready(self):
        self.get_rostopic = None
        while self.get_rostopic is None and not rospy.is_shutdown():
            try:
                self.get_rostopic = rospy.wait_for_message(
                    "/LeftArm/Position", ArmMsgs, timeout=1.0)
                rospy.logdebug(
                    "Current "+"/LeftArm/Position "+" READY=>" + str(self.get_joint_states))
            except:
                rospy.logerr(
                    "Current "+"/LeftArm/Position "+" not ready yet, retrying....")
        return self.get_rostopic

    # def set_all_joint_position(self, joint_angle):
    #     self.set_joint_states.header.stamp = rospy.Time.now()
    #     self.set_joint_states.name = self.joint_state_name
    #     self.set_joint_states.position = joint_angle
    #     self.js_pub.publish(self.set_joint_states)
    #     return True

    def set_left_arm_position(self, joint_angle):
        for i in range(7):
            if joint_angle[i] > self.left_arm_upper_limit[i] or joint_angle[i] < self.left_arm_lower_limit[i]:
                return False
        if self.use_sim_env:
            self.left_arm_joint_pub.publish(self.joint_2_armmsg(joint_angle))
        else:
            self.get_left_arm_joint = joint_angle
        return True

    def set_right_arm_position(self, joint_angle):
        for i in range(7):
            if joint_angle[i] > self.right_arm_upper_limit[i] or joint_angle[i] < self.right_arm_lower_limit[i]:
                return False
        if self.use_sim_env:
            self.right_arm_joint_pub.publish(self.joint_2_armmsg(joint_angle))
        else:
            self.get_right_arm_joint = joint_angle
        return True

    def set_main_body_position(self, joint_angle):
        for i in range(4):
            if joint_angle[i] > self.main_body_upper_limit[i] or joint_angle[i] < self.main_body_lower_limit[i]:
                return False
        if self.use_sim_env:
            self.main_body_joint_pub.publish(self.joint_2_body(joint_angle))
        else:
            self.get_main_body_joint = joint_angle
        return True

    def set_head_body_position(self, joint_angle):
        if self.use_sim_env:
            self.head_body_joint_pub.publish(self.joint_2_head(joint_angle))
        else:
            self.get_head_body_joint = joint_angle
        return True

    def set_left_hand_position(self, joint_angle):
        if self.use_sim_env:
            self.left_hand_joint_pub.publish(self.joint_2_handmsg(joint_angle))
        else:
            self.get_left_hand_joint = joint_angle
        return True

    def set_right_hand_position(self, joint_angle):
        if self.use_sim_env:
            self.right_hand_joint_pub.publish(self.joint_2_handmsg(joint_angle))
        else:
            self.get_right_hand_joint = joint_angle
        return True

    # def get_all_joint_position(self):
    #     return self.get_joint_states.position

    def get_left_arm_position(self):
        if self.use_sim_env:
            return self.armmsg_2_joint(self.left_arm_joint)
        else:
            return self.get_left_arm_joint

    def get_right_arm_position(self):
        if self.use_sim_env:
            return self.armmsg_2_joint(self.right_arm_joint)
        else:
            return self.get_right_arm_joint

    def get_main_body_position(self):
        if self.use_sim_env:
            return self.body_2_joint(self.main_body_joint)
        else:
            return self.get_main_body_joint

    def get_head_body_position(self):
        if self.use_sim_env:
            return self.head_2_joint(self.head_body_joint)
        else:
            return self.get_head_body_joint

    def get_left_hand_position(self):
        if self.use_sim_env: 
            return self.handmsg_2_joint(self.left_hand_joint)
        else:
            return self.get_left_hand_joint

    def get_right_hand_position(self):
        if self.use_sim_env: 
            return self.handmsg_2_joint(self.right_hand_joint)
        else:
            return self.get_right_hand_joint

    # def get_ee_pose(self):
    #     # todo

    # def sub_js_callback(self, msg):
    #     self.get_joint_states = msg

    def sub_main_body_joint_callback(self, msg):
        self.main_body_joint = copy.copy(msg)

    def sub_head_body_joint_callback(self, msg):
        self.head_body_joint = copy.copy(msg)

    def sub_left_arm_joint_callback(self, msg):
        self.left_arm_joint = copy.copy(msg)

    def sub_right_arm_joint_callback(self, msg):
        self.right_arm_joint = copy.copy(msg)

    def sub_left_hand_joint_callback(self, msg):
        self.left_hand_joint = copy.copy(msg)

    def sub_right_hand_joint_callback(self, msg):
        self.right_hand_joint = copy.copy(msg)

    def get_joint_names(self):
        return self.get_joint_states.name

    def joint_2_armmsg(self, joint):
        arm_msg = ArmMsgs()
        arm_msg.Shoulder_X = joint[0]
        arm_msg.Shoulder_Y = joint[1]
        arm_msg.Elbow_Z = joint[2]
        arm_msg.Elbow_X = joint[3]
        arm_msg.Wrist_Z = joint[4]
        arm_msg.Wrist_X = joint[5]
        arm_msg.Wrist_Y = joint[6]
        return arm_msg

    def joint_2_handmsg(self, joint):
        hand_msg = HandMsgs()
        hand_msg.Thumb = joint[0]
        hand_msg.Index = joint[1]
        hand_msg.Middle = joint[2]
        hand_msg.Ring = joint[3]
        hand_msg.Pinky = joint[4]
        return hand_msg

    def joint_2_body(self, joint):
        body_msg = BodyMsgs()
        body_msg.Knee = joint[0]
        body_msg.Back_Z = joint[1]
        body_msg.Back_X = joint[2]
        body_msg.Back_Y = joint[3]
        return body_msg

    def joint_2_head(self, joint):
        head_msg = HeadMsgs()
        head_msg.Neck_Z = joint[0]
        head_msg.Neck_X = joint[1]
        head_msg.Head = joint[2]
        return head_msg

    def armmsg_2_joint(self, msg):
        joint = [msg.Shoulder_X, msg.Shoulder_Y,
                 msg.Elbow_Z, msg.Elbow_X,
                 msg.Wrist_Z, msg.Wrist_X, msg.Wrist_Y]
        return joint

    def handmsg_2_joint(self, msg):
        joint = [msg.Thumb, msg.Index, msg.Middle, msg.Ring, msg.Pinky]
        return joint

    def body_2_joint(self, msg):
        joint = [msg.Knee,
                 msg.Back_Z, msg.Back_X, msg.Back_Y]
        return joint

    def head_2_joint(self, msg):
        joint = [msg.Neck_Z, msg.Neck_X, msg.Head]
        return joint

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
