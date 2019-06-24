#!/usr/bin/env python
# coding: utf-8

import math, random
import time
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from spinup.utils.logx import EpochLogger

start_time = time.time()
import os
import psutil
import sys
process = psutil.Process(os.getpid())

class EfficientReplayBuffer:
    """
    Enhanced replay buffer that uses memory more efficiently
    works for discrete action space
    """
    def __init__(self, obs_dim, size):
        """
        :param obs_dim: size of observation
        :param act_dim: size of the action
        :param size: size of the buffer
        """
        # ## init buffers as numpy arrays
        # self.obs_buf = np.zeros([size]+list(obs_dim), dtype=np.uint8)
        # self.acts_buf = np.zeros(size, dtype=np.uint8)
        # self.rews_buf = np.zeros(size, dtype=np.float32)
        # # by convention this done is whether next state is terminal
        # self.done_buf = np.zeros(size, dtype=np.uint8)

        ## init buffers as numpy arrays, init to ones allow us to allocate memory
        ## during init so it becomes easier to find problems
        self.obs_buf = np.ones([size] + list(obs_dim), dtype=np.uint8)
        self.acts_buf = np.ones(size, dtype=np.uint8)
        self.rews_buf = np.ones(size, dtype=np.float32)
        # by convention this done is whether next state is terminal
        self.done_buf = np.ones(size, dtype=np.uint8)

        # this is wether current state is terminal (which means it doesn't have a following state)
        # we need this to help make memory usage more efficient
        # need to init it to ones to prevent problems when buffer is not full
        self.current_state_done_buf = np.ones(size, dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, done, current_done):
        """
        data will get stored in the pointer's location
        data should NOT be in tensor format.
        it's easier if you get data from environment
        then just store them with the given format
        """
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.current_state_done_buf[self.ptr] = current_done
        ## move the pointer to store in next location in buffer
        self.ptr = (self.ptr+1) % self.max_size
        ## keep track of the current buffer size
        self.size = min(self.size+1, self.max_size)

    def __len__(self):
        return self.size

    def sample_uniform_batch(self, batch_size):
        ## sample with replacement from buffer
        idxs = np.random.randint(0, self.size, size=batch_size)

        for i in range(len(idxs)):
            while self.current_state_done_buf[idxs[i]] == 1:
                # if == 1 means this state is a terminal state
                # then need to resample this one
                idxs[i] = np.random.randint(0, self.size)

        next_state_idxs = (idxs + 1) % self.size

        return dict(obs1=self.obs_buf[idxs],
                    obs2=self.obs_buf[next_state_idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon, env, test_action=False):
        # if evaluation action is True then always take greedy action
        epsilon_to_use = 0.05 if test_action else epsilon
        if random.random() > epsilon_to_use:
            with torch.no_grad():
                state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0))
                q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action

# <h3>Synchronize current policy net and target net</h3>

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def test_agent(n, current_model, test_env):
    ep_return_list = np.zeros(n)
    for j in range(n):
        o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
        while not d:
            # Take deterministic actions at test time
            a = current_model.act(state=o, epsilon=0, env=test_env, test_action=True)
            o, r, d, _ = test_env.step(a)
            ep_ret += r
            ep_len += 1
        ep_return_list[j] = ep_ret
        logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

def get_current_epsilon(current_timestep):
    # given the current timestep (number of step, not number of frame), output the current epsilon value
    decay_total_timestep = int(1e6)
    epsilon_start = 1.0
    epsilon_final = 0.1
    if current_timestep < decay_total_timestep:
        current_epsilon = (current_timestep/decay_total_timestep)*epsilon_final + \
                          (1-current_timestep/decay_total_timestep)*epsilon_start
        return current_epsilon
    else:
        return epsilon_final

### Important: the whole trainning model

def double_dqn(env_id ="PongNoFrameskip-v4", seed = 0, buffer_size = int(1e5), replay_initial = 50000,
               num_step = int(5e7), batch_size = 32, lr = 0.00025, gamma = 0.99, tau=10000, test_freq=int(1e5),
               logger_kwargs=dict()):
    """check device"""
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        print("Using cuda")
    else:
        print("Using cpu")
    global Variable
    Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else \
        autograd.Variable(*args, **kwargs)
    """set up logger"""
    global logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    """wrap up atari environment"""
    from common.wrappers import NoopResetEnv, wrap_deepmind, wrap_pytorch
    env, test_env = gym.make(env_id), gym.make(env_id)
    test_env = NoopResetEnv(test_env, noop_max=30) # no-op only applies to test env
    env, test_env = wrap_deepmind(env, frame_stack=True), wrap_deepmind(test_env, frame_stack=True)
    env, test_env = wrap_pytorch(env), wrap_pytorch(test_env)
    ## seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)
    ## seed environment along with env action space so that everything about env is seeded
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    test_env.seed(seed)
    test_env.action_space.np_random.seed(seed)

    # Training setup
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n
    print('Processed observation space', obs_dim, 'action space (%s)' % act_dim)
    sys.stdout.flush()

    replay_buffer = EfficientReplayBuffer(obs_dim=obs_dim, size=buffer_size)

    print("mem in gb after replay buffer init:", process.memory_info().rss / 1e9)  # in gb

    current_model = CnnDQN(env.observation_space.shape, env.action_space.n)
    target_model = CnnDQN(env.observation_space.shape, env.action_space.n)
    if USE_CUDA:
        current_model = current_model.cuda()
        target_model = target_model.cuda()

    optimizer = optim.Adam(current_model.parameters(), lr=lr)
    huber_criterion = nn.SmoothL1Loss()

    update_target(current_model, target_model)

    # <h2>Epsilon greedy exploration</h2>

    print("mem in gb after replay buffer and model init:", process.memory_info().rss / 1e9)  # in gb
    sys.stdout.flush()
    epoch, episode_reward, ep_len = 0, 0, 0

    state = env.reset()
    for step_idx in range(1, num_step + 1):
        # epsilon-greedy explore
        epsilon = get_current_epsilon(step_idx)
        action = current_model.act(state, epsilon, env)
        next_state, reward, done, _ = env.step(action)
        # save a non-terminal state
        replay_buffer.store(state, action, reward, done, False)
        if done == True:
            # save a terminal state to efficient buffer
            replay_buffer.store(next_state, action, reward, done, True)

        state = next_state
        episode_reward += reward
        ep_len += 1

        if done:
            state = env.reset()
            logger.store(EpRet=episode_reward, EpLen=ep_len)
            episode_reward, ep_len = 0, 0

        '''According to the DDQN paper "The memory gets sampled to update the network every 4
        steps with minibatches of size 32'''
        if step_idx % 4 == 0 and len(replay_buffer) >= replay_initial: #TODO
            # get data from replay buffer
            batch = replay_buffer.sample_uniform_batch(batch_size)

            S = Variable(torch.FloatTensor(batch['obs1']))
            S_prime = Variable(torch.FloatTensor(batch['obs2']))
            A = Variable(torch.LongTensor(batch['acts']))
            # unsqueeze is to make sure rewards and done tensors are of the shape nx1, instead of n
            # to prevent problems later
            R = Variable(torch.FloatTensor(batch['rews']))
            D = Variable(torch.FloatTensor(batch['done']))

            q_values = current_model(S)
            next_q_values = current_model(S_prime)
            next_q_target_values = target_model(S_prime)

            q_value = q_values.gather(1, A.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_target_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

            expected_q_value = R + gamma * next_q_value * (1 - D)

            loss = huber_criterion(q_value, expected_q_value.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # store diagnostic info to logger
            if step_idx % 1000 == 0:
                q_mean = q_value.mean().cpu().item()
                logger.store(Loss=loss.cpu().item(), QVal=q_mean)

        if step_idx % tau == 0:
            update_target(current_model, target_model)

        if step_idx % test_freq == 0 and len(replay_buffer) >= replay_initial:
            epoch += 1
            print('epoch', epoch)
            # Test the agent
            test_agent(10, current_model, test_env)
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', step_idx)
            logger.log_tabular('QVal', with_min_and_max=True)
            logger.log_tabular('Loss', average_only=True)
            logger.log_tabular('Mem', (process.memory_info().rss / 1e9))
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--t_freq', type=int, default=int(1e5))
    parser.add_argument('--exp_name', type=str, default='ddqn')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--steps', type=int, default=int(5e7))
    parser.add_argument('--b_size', type=int, default=int(1e5))
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--replay_init', type=int, default=50000)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    double_dqn(env_id=args.env, seed=args.seed, buffer_size=args.b_size, replay_initial=args.replay_init, num_step=args.steps,
               batch_size=args.batch, test_freq=args.t_freq, logger_kwargs=logger_kwargs)

