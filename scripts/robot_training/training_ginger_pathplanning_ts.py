#!/usr/bin/env python3
import os
import gym
import torch
import pprint
import threading
import rospy
import rospkg
import argparse
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import VectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from robot_training.util.utility import EnvRegister


class Net(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), 512),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(512, 512), nn.ReLU(inplace=True)]
        if action_shape:
            self.model += [nn.Linear(512, np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, state


def thread_job():
    rospy.spin()


def get_args():
    rospackage_name = "robot_training"
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    log_path = pkg_path + '/learning_result/log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        rospy.loginfo("Created folder="+str(log_path))

    rospy.loginfo("read parm from yaml")
    task_env_name = rospy.get_param(
        '/ginger_training/task_and_robot_environment_name')
    lr = rospy.get_param('/ginger_training/lr')
    # lr_decay = rospy.get_param('/ginger_training/lr_decay') torch.optim.Adam donot have this parm?
    gamma = rospy.get_param('/ginger_training/gamma')
    replay_buffer_size = rospy.get_param('/ginger_training/replay_buffer_size')
    batch_size = rospy.get_param('/ginger_training/batch_size')
    epsilon_training = rospy.get_param('/ginger_training/epsilon_training')
    epsilon_running = rospy.get_param('/ginger_training/epsilon_running')
    epsilon_decay = rospy.get_param('/ginger_training/epsilon_decay')
    epsilon_min = rospy.get_param('/ginger_training/epsilon_min')
    seed = rospy.get_param('ginger_training/seed')
    layer_num = rospy.get_param('ginger_training/layer_num')
    training_num = rospy.get_param('ginger_training/training_num')
    test_num = rospy.get_param('ginger_training/test_num')
    epoch = rospy.get_param('ginger_training/epoch')
    step_per_epoch = rospy.get_param('ginger_training/step_per_epoch')
    estimation_step = rospy.get_param('ginger_training/estimation_step')
    target_update_freq = rospy.get_param('ginger_training/target_update_freq')
    collect_per_step = rospy.get_param('ginger_training/collect_per_step')
    render = rospy.get_param('ginger_training/render')

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=task_env_name)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--eps-test', type=float, default=epsilon_running)
    parser.add_argument('--eps-train', type=float, default=epsilon_training)
    parser.add_argument('--eps-decay', type=float, default=epsilon_decay)
    parser.add_argument('--eps-min', type=float, default=epsilon_min)
    parser.add_argument('--buffer-size', type=int, default=replay_buffer_size)
    parser.add_argument('--batch-size', type=int, default=batch_size)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--gamma', type=float, default=gamma)
    parser.add_argument('--training-num', type=int, default=training_num)
    parser.add_argument('--test-num', type=int, default=test_num)
    parser.add_argument('--epoch', type=int, default=epoch)
    parser.add_argument('--step-per-epoch', type=int, default=step_per_epoch)
    parser.add_argument('--layer-num', type=int, default=layer_num)
    parser.add_argument('--logdir', type=str, default=log_path)
    parser.add_argument('--n-step', type=int, default=estimation_step)
    parser.add_argument('--target-update-freq', type=int,
                        default=target_update_freq)
    parser.add_argument('--collect-per-step', type=int,
                        default=collect_per_step)
    parser.add_argument('--render', type=float, default=render)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def test_dqn(args=get_args()):
    # env
    task_env = EnvRegister(args.task)
    env = gym.make(task_env)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    rospy.loginfo(args.state_shape)
    rospy.loginfo(args.action_shape)
    train_envs = VectorEnv(
        [lambda: gym.make(task_env) for _ in range(args.training_num)])
    test_envs = VectorEnv(
        [lambda: gym.make(task_env) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(args.layer_num, args.state_shape, args.action_shape, args.device)
    net = net.to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(
        net, optim, args.gamma, args.n_step,
        use_target_network=args.target_update_freq > 0,
        target_update_freq=args.target_update_freq)
    # collector
    rospy.loginfo("init collector")
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=args.batch_size)
    # log
    writer = SummaryWriter(args.logdir + '/' + 'dqn')

    rew_record = []

    def stop_fn(x):
        if x >= 10000:
            rew_record.append(x)
            if(len(rew_record) > 100):
                return True
            else:
                return False
        else:
            rew_record.clear()
            return False

    def train_fn(x):
        policy.set_eps(args.eps_train, args.eps_decay, args.eps_min)

    def test_fn(x):
        policy.set_eps(args.eps_test, args.eps_decay, args.eps_min)

    # trainer
    rospy.loginfo("start training")
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, writer=writer)

    assert stop_fn(result['best_reward'])
    pprint.pprint(result)
    train_collector.close()
    test_collector.close()
    # save network
    torch.save(net, 'ginger_dqn_pathplanning.pkl')

    rospy.loginfo("training finish, testing...")
    # Let's watch its performance!
    env_test = gym.make(task_env)
    net_test = torch.load('ginger_dqn_pathplanning.pkl')
    policy_test = DQNPolicy(
        net_test, optim, args.gamma, args.n_step,
        use_target_network=args.target_update_freq > 0,
        target_update_freq=args.target_update_freq)
    collector = Collector(policy_test, env_test)
    result = collector.collect(n_episode=1, render=args.render)
    rospy.loginfo(f'Final reward: {result["rew"]}, length: {result["len"]}')
    collector.close()


if __name__ == '__main__':
    rospy.init_node('ginger_training_pathplanning_ts',
                    anonymous=True, log_level=rospy.INFO)

    # add thread, deal with ros callback function
    add_thread = threading.Thread(target=thread_job)
    add_thread.start()
    rospy.sleep(1)

    test_dqn(get_args())

    rospy.spin()
