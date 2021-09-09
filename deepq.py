import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import argparse

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
#import numpy as np
import collections


# class DQN(nn.Module):
#     def __init__(self, input_shape, n_actions):
#         super(DQN, self).__init__()
#         self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#                                 nn.ReLU(),
#                                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
#                                 nn.ReLU(),
#                                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
#                                 nn.ReLU()
#                                 )
#         conv_out_size = self._get_conv_out(input_shape)
#         self.fc = nn.Sequential(nn.Linear(conv_out_size, 512),
#                                 nn.ReLU(),
#                                 nn.Linear(512, n_actions)
#                                 )
#     def _get_conv_out(self, shape):        
#         o = self.conv(torch.zeros(1, *shape))
#         print(o)
#         return int(np.prod(o.size()))     
#     def forward(self, x):
#         conv_out = self.conv(x).view(x.size()[0], -1) # view function reshapes the tensor
#         return self.fc(conv_out)   

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        # self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        #                         nn.ReLU(),
        #                         nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #                         nn.ReLU(),
        #                         nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #                         nn.ReLU()
        #                         )
        # conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(nn.Linear(input_shape[0], 8),
                                nn.ReLU(),
                                nn.Linear(8, n_actions)
                                )
    def _get_conv_out(self, shape):        
        o = self.conv(torch.zeros(1, *shape))
        print(o)
        return int(np.prod(o.size()))     
    def forward(self, x):
        #conv_out = self.conv(x).view(x.size()[0], -1) # view function reshapes the tensor
        return self.fc(x.float())           

env = gym.make('CartPole-v0').unwrapped  
DEFAULT_ENV_NAME= 'CartPole-v0'
MEAN_REWARD_BOUND = 160
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000    
EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02     
Experience = collections.namedtuple('Experience', field_names=
['state', 'action', 'reward', 'done', 'new_state'])     


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def __len__(self):
        return len(self.buffer)
    def append(self, experience):
        self.buffer.append(experience)
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                        replace=False)
        states, actions, rewards, dones, next_states = zip(*
        [self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(next_states)
class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()
    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0            
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())       
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        new_state = new_state
        exp = Experience(self.state, action, reward, is_done,
                            new_state)  
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward                           
def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch     
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)   
    done_mask = torch.ByteTensor(dones).to(device)
    state_action_values = net(states_v).gather(1,
    actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    #next_state_values = next_state_value.detach()
    next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
    action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
        help="Name of the environment, default=" +
        DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float,
    default=MEAN_REWARD_BOUND,
    help="Mean reward boundary for stop of   training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")    
    net = DQN(env.observation_space.shape,
    env.action_space.n).to(device)
    tgt_net = DQN(env.observation_space.shape,
    env.action_space.n).to(device)
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    while True:
        frame_idx += 1
        print(frame_idx)
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx /
                            EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon, device=device)
        mean_reward = np.mean(total_rewards[-100:])   
        if best_mean_reward is not None:
            print("Best mean reward updated %.3f ->%.3f, model saved" % (best_mean_reward, mean_reward))
            best_mean_reward = mean_reward
        if mean_reward > args.reward:
            print("Solved in %d frames!" % frame_idx)
            break
        if len(buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())  
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()                                   