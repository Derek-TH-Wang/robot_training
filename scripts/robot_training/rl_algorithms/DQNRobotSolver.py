# Inspired by https://keon.io/deep-q-learning/
import time
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import model_from_yaml
import rospkg
import rospy
import os
import matplotlib.pyplot as plt
#from keras.callbacks import TensorBoard


class DQNRobotSolver():
    def __init__(self, env, environment_name, n_observations, n_actions, n_win_ticks=195, min_episodes=100, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, reached_goal_reward=100, alpha=0.01, alpha_decay=0.01, batch_size=64, replay_buffer_size=3000, monitor=False, quiet=False):
        self._env = env
        if monitor:
            rospy.loginfo("monitor")
            rospackage_name = "robot_training"
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path(rospackage_name)
            outdir = pkg_path + '/learning_result/data/' + environment_name
            if not os.path.exists(outdir):
                os.makedirs(outdir)
                rospy.loginfo("Created folder="+str(outdir))
            self._env = gym.wrappers.Monitor(self._env, outdir, force=True)

        self.memory = deque(maxlen=replay_buffer_size)
        self.input_dim = n_observations
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_win_ticks = n_win_ticks
        self.min_episodes = min_episodes
        self.batch_size = batch_size
        self.quiet = quiet
        self.reached_goal_reward = reached_goal_reward
        if max_env_steps is not None:
            self._env._max_episode_steps = max_env_steps

        # Init model
        #self.model_name = "dqn_ginger_pathplanning"
        #self.tensorboard = TensorBoard(log_dir=pkg_path + '/learning_result/logs/{}'.format(self.model_name))
        self.model = Sequential()
        self.model.add(
            Dense(256, input_dim=self.input_dim, activation='tanh'))
        self.model.add(Dense(1024, activation='tanh'))
        self.model.add(Dense(4096, activation='tanh'))
        self.model.add(Dense(self.n_actions, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(
            lr=self.alpha, decay=self.alpha_decay))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon, do_train, iteration=0):

        if do_train and (np.random.random() <= epsilon):
            # We return a random sample form the available action space
            action_chosen = self._env.action_space.sample()
            rospy.logwarn(">>>>>Chosen Random ACTION = " + str(action_chosen))
        else:
            # We return the best known prediction based on the state
            action_chosen = np.argmax(self.model.predict(state))
            rospy.logwarn(">>>>>Chosen Predict ACTION = " + str(action_chosen))

        if do_train:
            rospy.logdebug("LEARNING A="+str(action_chosen) +
                           ",E="+str(round(epsilon, 3))+",I="+str(iteration))
        else:
            rospy.logdebug("RUNNING A="+str(action_chosen) +
                           ",E="+str(round(epsilon, 3))+",I="+str(iteration))

        return action_chosen

    def get_epsilon(self, t):
        # new_epsilon = max(self.epsilon_min, min(
        #    self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))
        new_epsilon = self.epsilon
        return new_epsilon

    def preprocess_state(self, state):
        return np.reshape(state, [1, self.input_dim])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            #rospy.logwarn("action = " + str(action))
            # rospy.logfatal(y_target)
            y_target[0][action] = reward if done else reward + \
                self.gamma * np.max(self.model.predict(next_state)[0])
            # rospy.logfatal(y_target)
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        # history = self.model.fit(np.array(x_batch), np.array(y_batch),
        #                         batch_size=len(x_batch), verbose=0, callbacks=[self.tensorboard])
        history = self.model.fit(np.array(x_batch), np.array(y_batch),
                                 batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        rospy.logwarn("loss = " + str(history.history['loss']))

    def run(self, num_episodes, do_train=False):

        scores = deque(maxlen=100)

        # for e in range(num_episodes):
        e = 0
        while not rospy.is_shutdown():
            rospy.loginfo("-----------------------------------------")

            init_state = self._env.reset()
            state = self.preprocess_state(init_state)
            done = False
            i = 0
            reward_in_episode = 0
            while not done:
                # openai_ros doesnt support render for the moment
                # self._env.render()
                action = self.choose_action(
                    state, self.get_epsilon(e), do_train, i)
                next_state, reward, done, reach_goal = self._env.step(action)
                next_state = self.preprocess_state(next_state)
                if do_train:
                    # If we are training we want to remember what I did and process it.
                    self.remember(state, action, reward, next_state, done)
                reward_in_episode += reward
                state = next_state
                i += 1

            rospy.logwarn("reward_in_episode = " + str(reward_in_episode))
            scores.append(reward_in_episode)
            if min(scores) > self.reached_goal_reward:
                rospy.logfatal("reach goal, training finish")
                return e

            if do_train:
                self.replay(self.batch_size)
            e = e+1

        # if not self.quiet:
        #     rospy.logfatal('Did not solve after {} episodes'.format(e))
        return e

    def save(self, model_name, models_dir_path="/tmp"):
        """
        We save the current model
        """

        model_name_yaml_format = model_name+".yaml"
        model_name_HDF5_format = model_name+".h5"

        model_name_yaml_format_path = os.path.join(
            models_dir_path, model_name_yaml_format)
        model_name_HDF5_format_path = os.path.join(
            models_dir_path, model_name_HDF5_format)

        # serialize model to YAML
        model_yaml = self.model.to_yaml()

        with open(model_name_yaml_format_path, "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5: http://www.h5py.org/
        self.model.save_weights(model_name_HDF5_format_path)
        rospy.logwarn("Saved model to disk")

    def load(self, model_name, models_dir_path="/tmp"):
        """
        Loads a previously saved model
        """

        model_name_yaml_format = model_name+".yaml"
        model_name_HDF5_format = model_name+".h5"

        model_name_yaml_format_path = os.path.join(
            models_dir_path, model_name_yaml_format)
        model_name_HDF5_format_path = os.path.join(
            models_dir_path, model_name_HDF5_format)

        # load yaml and create model
        yaml_file = open(model_name_yaml_format_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self.model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        self.model.load_weights(model_name_HDF5_format_path)
        rospy.logwarn("Loaded model from disk")
