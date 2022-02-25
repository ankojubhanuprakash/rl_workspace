from typing import Collection
from numpy.core.numeric import indices
import torch 
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
#from torch._C import T
#from torch.nn.modules.conv import ConvTranspose2d
#import torchvision
#from torchvision.transforms import ToTensor, Normalize, Compose
#from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import collections
import argparse
import time


def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("WARNING: For this notebook to perform best, "
        "if possible, in the menu under `Runtime` -> "
        "`Change runtime type.`  select `GPU` ")
  else:
    print("GPU is enabled in this notebook.")

  return device
  
def set_seed(seed=None, seed_torch=True):
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')


# In case that `DataLoader` is used
def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

SEED = 2021
set_seed(seed=SEED)
DEVICE = set_device()

class DQN_policy(nn.Module):
    def __init__(self,obssapce,n_action):
        super(DQN_policy,self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(obssapce,2 ),
            nn.ReLU(),
            nn.Linear(2,n_action)
                )
    def forward(self,x):
        
        
        return self.fc(x)


# Hyper Parameters
GAMMA =0.99
BATCH_SIZE=99
REPLAY_SIZE = 100000
MEAN_REWARD_BOUND = 20
REPLAY_START_SIZE =1000
LEARNING_RATE= 1e-3
EPS_DCY = 0.999999
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
SYNC_TARGET_FRAMES = 100

#replay buffer
Experience = collections.namedtuple('Experience',field_names=['state','action','reward','done','new_state'])

class ExperienceBuffer:
    def __init__(self,capacity):
        self.buffer = collections.deque()

    def __len__(self):
        return(len(self.buffer))
    def append(self,experince):
        self.buffer.append(experince)
    def sample(self,batch_size):
        idxs = np.random.choice(len(self.buffer),batch_size,replace=False)
        states,actions,rewards,dones,next_states = zip(*[self.buffer[idx] for idx in idxs ])
        return np.array(states),np.array(actions),np.array(rewards),np.array(dones),np.array(next_states)

class Agent():
    def __init__(self,env,exp_buffer) :
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()
    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0
    def play_step(self,net,epsilon=0.0,device=DEVICE):
        #print('hi')
        done_reward = None
        if np.random.random()<epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state],copy=False)
            state_v = torch.tensor(state_a,dtype=torch.float32).to(device)
            #clearprint(state_v.dtype)
            q_vals_v = net(state_v)
            act_v = torch.argmax(q_vals_v)
            action = int(act_v.item())
        new_state, reward, is_done, _ = self.env.step(action)

        self.total_reward+=reward
        #new_state =     
        exp = Experience(self.state,action, reward, is_done, new_state)
        #print(type(self.exp_buffer),exp)
        
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
          done_reward = self.total_reward
          self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states,dtype=torch.float32).to(device)
    next_states_v = torch.tensor(next_states,dtype=torch.float32).to(device)
    actions_v = torch.tensor(actions,dtype=torch.int64).to(device)
    rewards_v = torch.tensor(rewards,dtype=torch.float32).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    DEFAULT_ENV_NAME = 'MountainCar-v0'
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--mode", default=True, action="store_true", help="play mode")
                        
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = DEVICE

    env = gym.make(args.env)
    if args.mode:
        net = DQN_policy(env.observation_space.shape[0], env.action_space.n)
        net.load_state_dict(torch.load('MountainCar-v0-best.dat', map_location=lambda stg, _: stg))
        stat= env.reset()
        while True:
            q_vals_v = net(torch.tensor(stat,dtype=torch.float32))
            act_v = torch.argmax(q_vals_v)
            action = int(act_v.item())
            #action = 
            stat, reward, is_done, _ = env.step(action)
            env.render()

            if is_done:
                break
            

    else:     
        net = DQN_policy(env.observation_space.shape[0], env.action_space.n).to(device)
        tgt_net = DQN_policy(env.observation_space.shape[0], env.action_space.n).to(device)
        writer = SummaryWriter(comment="-" + args.env)
        #print(net)

        buffer = ExperienceBuffer(REPLAY_SIZE)
        agent = Agent(env, buffer)
        epsilon = EPSILON_START

        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
        total_rewards = []
        frame_idx = 0
        ts_frame = 0
        ts = time.time()
        best_mean_reward = None
        Episodes=10000
        epsilon = EPSILON_START

        #for episode in range(epEisode):
        flag=0

        while True:
            frame_idx += 1
            flag+=1
            epsilon = epsilon*EPS_DCY#max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
            #print(epsilon)
            reward = agent.play_step(net, epsilon, device=device)
            if reward is not None:
                total_rewards.append(reward)
                #speed = (frame_idx - ts_frame) / (time.time() - ts)
                #ts_frame = frame_idx
                #ts = time.time()
                mean_reward = np.mean(total_rewards[-100:])
                print(" %d games, mean reward %.3f, eps %.2f, " % (
                    len(total_rewards), mean_reward, epsilon,
                    
                ))
                writer.add_scalar("epsilon", epsilon, frame_idx)
                #writer.add_scalar("speed", speed, frame_idx)
                writer.add_scalar("reward_100", mean_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)
                if best_mean_reward is None or best_mean_reward < mean_reward:
                    torch.save(net.state_dict(), args.env + "-best.dat")
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward
                if mean_reward > args.reward:
                    print("Solved in %d frames!" % frame_idx)
                    break
            #print(len(buffer))        
            if flag>=1000:# < REPLAY_START_SIZE:
                flag=0
            else:    
                continue

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())
            #print('trainig',frame_idx)
            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            loss_t.backward()
            optimizer.step()
        writer.close() 