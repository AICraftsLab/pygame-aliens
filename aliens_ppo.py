# -*- coding: utf-8 -*-
"""torch_ppo.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eAdSFuqxV9fkmjpQyS19J20BlCszMggz
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import warnings
from torch.distributions.categorical import Categorical
import pygame as pg

from nn_visualizer import NeuralNetworkVisualizer
from aliens import AliensEnv

warnings.simplefilter("ignore")

class PPOMemory():
    """
    Memory for PPO
    """
    def  __init__(self, batch_size):
        self.states = []
        self.actions= []
        self.action_probs = []
        self.rewards = []
        self.vals = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        ## suppose n_states=20 and batch_size = 4
        n_states = len(self.states)
        ##n_states should be always greater than batch_size
        ## batch_start is the starting index of every batch
        ## eg:   array([ 0,  4,  8, 12, 16]))
        batch_start = np.arange(0, n_states, self.batch_size)
        ## random shuffling if indexes
        # eg: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
        indices = np.arange(n_states, dtype=np.int64)
        ## eg: array([12, 17,  6,  7, 10, 11, 15, 13, 18,  9,  8,  4,  3,  0,  2,  5, 14,19,  1, 16])
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        ## eg: [array([12, 17,  6,  7]),array([10, 11, 15, 13]),array([18,  9,  8,  4]),array([3, 0, 2, 5]),array([14, 19,  1, 16])]
        return np.array(self.states),np.array(self.actions),\
               np.array(self.action_probs),np.array(self.vals),np.array(self.rewards),\
               np.array(self.dones),batches




    def store_memory(self,state,action,action_prob,val,reward,done):
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.vals.append(val)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions= []
        self.action_probs = []
        self.rewards = []
        self.vals = []
        self.dones = []

class ActorNwk(nn.Module):
    def __init__(self,input_dim,out_dim,
                 adam_lr,
                 chekpoint_file,
                 hidden_dim=256
                 ):
        super(ActorNwk, self).__init__()

        self.actor_nwk = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim),
            nn.Softmax(dim=-1)
        )

        self.checkpoint_file = chekpoint_file
        self.optimizer = torch.optim.Adam(params=self.actor_nwk.parameters(),lr=adam_lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self,state):
        out = self.actor_nwk(state)
        dist = Categorical(out)
        return dist

    def save_checkpoint(self, episode):
        torch.save(self.state_dict(), self.checkpoint_file + episode)

    def load_checkpoint(self, episode):
        self.load_state_dict(torch.load(self.checkpoint_file + episode))

class CriticNwk(nn.Module):
    def __init__(self,input_dim,
                 adam_lr,
                 chekpoint_file,
                 hidden_dim=256
                 ):
        super(CriticNwk, self).__init__()

        self.critic_nwk = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),

        )

        self.checkpoint_file = chekpoint_file
        self.optimizer = torch.optim.Adam(params=self.critic_nwk.parameters(),lr=adam_lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self,state):
        out = self.critic_nwk(state)
        return out

    def save_checkpoint(self, episode):
        torch.save(self.state_dict(), self.checkpoint_file + episode)

    def load_checkpoint(self, episode):
        self.load_state_dict(torch.load(self.checkpoint_file + episode))

class Agent():
    def __init__(self, gamma, policy_clip,lamda, adam_lr,
                 n_epochs, batch_size, state_dim, action_dim):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.lamda  = lamda
        self.n_epochs = n_epochs

        self.actor = ActorNwk(input_dim=state_dim,out_dim=action_dim,adam_lr=adam_lr,chekpoint_file='./saves/actor')
        self.critic = CriticNwk(input_dim=state_dim,adam_lr=adam_lr,chekpoint_file='./saves/critic')
        self.memory = PPOMemory(batch_size)

    def store_data(self,state,action,action_prob,val,reward,done):
        self.memory.store_memory(state,action,action_prob,val,reward,done)


    def save_models(self, episode=''):
        print('... Saving Models ......')
        self.actor.save_checkpoint(episode)
        self.critic.save_checkpoint(episode)

    def load_models(self, episode=''):
        print('... Loading models ...')
        self.actor.load_checkpoint(episode)
        self.critic.load_checkpoint(episode)

    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.actor.device)

        dist = self.actor(state)
        ## sample the output action from a categorical distribution of predicted actions
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        
        ## value from critic model
        value = self.critic(state)
        value = torch.squeeze(value).item()

        return action, probs, value

    def calculate_advanatage(self,reward_arr,value_arr,dones_arr):
        time_steps = len(reward_arr)
        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        for t in range(0,time_steps-1):
            discount = 1
            running_advantage = 0
            for k in range(t,time_steps-1):
                if int(dones_arr[k]) == 1:
                    running_advantage += reward_arr[k] - value_arr[k]
                else:

                    running_advantage += reward_arr[k] + (self.gamma*value_arr[k+1]) - value_arr[k]

                running_advantage = discount * running_advantage
                # running_advantage += discount*(reward_arr[k] + self.gamma*value_arr[k+1]*(1-int(dones_arr[k])) - value_arr[k])
                discount *= self.gamma * self.lamda

            advantage[t] = running_advantage
        advantage = torch.tensor(advantage).to(self.actor.device)
        return advantage

    def learn(self):
        for _ in range(self.n_epochs):

            ## initially all will be empty arrays
            state_arr, action_arr, old_prob_arr, value_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            advantage_arr = self.calculate_advanatage(reward_arr,value_arr,dones_arr)
            values = torch.tensor(value_arr).to(self.actor.device)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage_arr[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage_arr[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage_arr[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()


if not os.path.exists('saves'):
    os.makedirs('saves')

episode_rewards = []

rewards_file_name = './saves/rewards.txt'

if os.path.exists(rewards_file_name):
    with open(rewards_file_name, 'r') as file:
        content = file.read()
        content = content.split(',')
        content = content[:-1]
        content = list(map(float, content))
        episode_rewards = content
else:
    print(f"The file {rewards_file_name} does not exist.")

rewards_file = open(rewards_file_name, 'a')
last_save_index = len(episode_rewards)

def save_lists_data():
    global last_save_index
    for reward in episode_rewards[last_save_index:]:
        rewards_file.write(str(reward)+',')
    rewards_file.flush()
    last_save_index = len(episode_rewards)

def standardize(state):
    data = np.array(state)

    # Compute the mean and standard deviation
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)

    # Avoid division by zero by setting std_dev to 1 where it is 0
    std_dev = np.where(std_dev == 0, 1, std_dev)

    # Standardize the data
    standardized_data = (data - mean) / std_dev

    return standardized_data

def plot_learning_curve(x, scores, figure_file=None):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    if figure_file:
        plt.savefig(figure_file)
    plt.show()

load_state = True
load_episode = 5150
surface = pg.display.set_mode((840, 660))
env = AliensEnv(surface, render=False, fps=100)
N = 1536
# batch_size = 5
batch_size = 256
n_epochs = 5
alpha = 0.0003
agent = Agent(state_dim=env.n_observations,
              action_dim=env.n_actions,
              batch_size=batch_size,
              n_epochs=n_epochs,
              policy_clip=0.2,
              gamma=0.99,lamda=0.95,
              adam_lr=alpha)
              
if load_state:
    agent.load_models(f'_{load_episode}')
    env.episode = load_episode
else:
    load_episode = 0
    
n_games = 5000
figure_file = './saves/learning_curve.png'
best_score = 17.58
learn_iters = 0
avg_score = 0
n_steps = 0
for i in range(load_episode, load_episode + n_games):
    state, info = env.reset()
    done = False
    episode_reward = 0
    while not done:
        state = standardize(state)
        action, prob, val = agent.choose_action(state)
        next_state, reward, terminated, info = env.step(action)
        # print(next_state, reward, terminated)
        done = 1 if terminated else 0
        n_steps += 1
        episode_reward += reward
        agent.store_data(state, action, prob, val, reward, done)
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        state = next_state
    episode_rewards.append(episode_reward)
    #episode_durations.append(n_steps)
    avg_score = round(np.mean(episode_rewards[-100:]), 2)
    if avg_score > best_score:
        best_score = avg_score
        print('Best Score')
        agent.save_models('_best')
    
    if i != load_episode and i % 25 == 0:
        save_lists_data()
        agent.save_models(f'_{i}')    
    
    print(info, 'Avg. reward', avg_score, 'N_Steps', n_steps, 'learn_iters', learn_iters)
    # print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            # 'time_steps', n_steps, 'learning_steps', learn_iters)

save_lists_data()
agent.save_models(f'_{load_episode + n_games}')  
#x = [i+1 for i in range(len(episode_rewards))]
#plot_learning_curve(x, episode_rewards, figure_file)

'''
from IPython.display import clear_output

env=AncientShooterEnv(render=False)


state_dim = env.n_observations
action_dim = env.n_actions

class ActorPredNwk(nn.Module):
    def __init__(self,input_dim,out_dim,
                 hidden1_dim=256,
                 hidden2_dim=256,
                 checkpoint_file = 'tmp/actor'
                 ):
        super(ActorPredNwk, self).__init__()

        self.checkpoint_file = checkpoint_file
        self.actor_nwk = nn.Sequential(
            nn.Linear(input_dim,hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim,hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim,out_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self,state):
        x = torch.Tensor(state)
        out = self.actor_nwk(x)
        return out

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


policy_nwk = ActorPredNwk(input_dim=state_dim,out_dim=action_dim)
policy_nwk.load_checkpoint()

for i in range(20):
  done = False
  state, info = env.reset()
  rewards = 0
  while not done:
      action = policy_nwk(state)
      action = torch.argmax(action).item()
      state, reward, done, info = env.step(action)
      if reward == 1:  # If enemy killed
          reward = 10  # Increase the reward
      rewards += reward
      print(state, reward, done, action)
      # print(current_state,new_state,action)
      # print('New state: ', new_state,'Reward: ',reward,'Terminated: ',terminated,'Truncated: ',truncated)
      # plt.title(f'Frame {i}')
      #plt.imshow(env.render())
      # plt.show()
  print(info, 'Reward: ',rewards)
'''