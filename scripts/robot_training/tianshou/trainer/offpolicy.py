import time
import tqdm
import rospy
import numpy as np

from tianshou.utils import tqdm_config, MovAvg
from tianshou.trainer import test_episode, gather_info


def offpolicy_trainer(policy, train_collector, test_collector, max_epoch,
                      step_per_epoch, collect_per_step, episode_per_test,
                      batch_size, train_fn=None, test_fn=None, stop_fn=None,
                      writer=None, log_interval=1, verbose=True, task=''):
    global_step = 0
    best_epoch, best_reward = -1, -1
    stat = {}
    start_time = time.time()
    for epoch in range(1, 1 + max_epoch):
        # train
        # 每个epoch，没有reset场景？？？？？
        policy.train()
        if train_fn:
            train_fn(epoch)
        with tqdm.tqdm(
                total=step_per_epoch, desc=f'Epoch #{epoch}',
                **tqdm_config) as t:
            while t.n < t.total:
                # if t.n == 0:
                #     reset_random_angle = True
                # else:
                #     reset_random_angle = False
                # result = train_collector.collect(n_step=collect_per_step, update_random_angle = reset_random_angle)
                result = train_collector.collect(n_step=collect_per_step)
                data = {}
                if stop_fn and stop_fn(result['info']):
                    test_result = test_episode(
                        policy, test_collector, test_fn,
                        epoch, episode_per_test)
                    if stop_fn and stop_fn(test_result['info']):
                        for k in result.keys():
                            data[k] = f'{result[k]:.2f}'
                        t.set_postfix(**data)
                        return gather_info(
                            start_time, train_collector, test_collector,
                            test_result['rew'])
                    else:
                        policy.train()
                        if train_fn:
                            train_fn(epoch)
                for i in range(min(
                        result['n/st'] // collect_per_step, t.total - t.n)):
                    global_step += 1
                    losses = policy.learn(train_collector.sample(batch_size))
                    for k in result.keys():
                        # donot print act, info
                        if type(result[k]) != type([]) and type(result[k]) != np.ndarray:
                            data[k] = f'{result[k]:.2f}'
                            if writer and global_step % log_interval == 0:
                                writer.add_scalar(
                                    k + '_' + task if task else k,
                                    result[k], global_step=global_step)
                    for k in losses.keys():
                        if stat.get(k) is None:
                            stat[k] = MovAvg()
                        stat[k].add(losses[k])
                        data[k] = f'{stat[k].get():.6f}'
                        if writer and global_step % log_interval == 0:
                            writer.add_scalar(
                                k + '_' + task if task else k,
                                stat[k].get(), global_step=global_step)
                    t.update(1)
                    t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # test
        result = test_episode(
            policy, test_collector, test_fn, epoch, episode_per_test)
        if best_epoch == -1 or best_reward < result['rew']:
            best_reward = result['rew']
            best_epoch = epoch
            best_act = result['all_act']
            rospy.logwarn(best_act)
        if verbose:
            print(f'Epoch #{epoch}: test_reward: {result["rew"]:.6f}, '
                  f'best_reward: {best_reward:.6f} in #{best_epoch}')
        print("test: ", result['info'])
        if stop_fn and stop_fn(result['info']):
            break
    return gather_info(
        start_time, train_collector, test_collector, best_reward)
