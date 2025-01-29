import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pygame as pg
from nn_visualizer import NeuralNetworkVisualizer
from aliens import AliensEnv

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256, 256)
        self.layer6 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        return self.layer6(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 150
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 30000
TAU = 0.005
LR = 1e-4

episode_rewards = []
episode_durations = []

surface = pg.display.set_mode((840, 660))
env = AliensEnv(surface, render=False)
env.fps = 100

# Get number of actions from gym action space
n_actions = env.n_actions
# Get the number of state observations
n_observations = env.n_observations

save_model = True
load_model = True
load_episode = 0
steps_done = 0

if load_model:
    load_episode = 1590
    steps_done = 981756
    env.episode = load_episode
    policy_net = torch.load(f'./saves2/aliens_policy_model_e{load_episode}_{steps_done}.pth')
    target_net = torch.load(f'./saves2/aliens_target_model_e{load_episode}_{steps_done}.pth')
else:
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(150000)  # 10000

#nn_viz = NeuralNetworkVisualizer(policy_net, surface)
saving_surface = pg.Surface((6000, 6000))
nn_saver = NeuralNetworkVisualizer(policy_net, saving_surface)

def get_eps_threshold():
    return EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = get_eps_threshold()
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def plot_rewards(moving_ave=40, show_result=False, save_path=None, save_only=False):
    """
    Plot the duration of episodes and optionally save the plot to a file.

    Parameters:
    - show_result (bool): If True, show the plot as the final result. Default is False.
    - save_path (str or None): Path to save the plot image. If None, the plot is not saved. Default is None.
    - save_only (bool): If True, save the plot but do not display it. Default is False.
    """
    # Create or activate the figure with ID 1
    plt.figure(1)

    # Convert list to a PyTorch tensor of type float
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    #eps_t = torch.tensor(episode_eps_thresholds, dtype=torch.float)

    # Set the title based on whether the result should be shown or not
    if show_result:
        plt.title('Result')
    else:
        # Clear the current figure if not showing the final result
        plt.clf()
        plt.title('Training...')

    # Label the x-axis and y-axis
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # Plot the episode durations
    plt.plot(rewards_t.numpy())

    # If there are at least moving_ave episodes, plot a moving average of the last 100 episodes
    if len(rewards_t) >= moving_ave:
        # Compute the moving average over a window of 100 episodes
        means = rewards_t.unfold(0, moving_ave, 1).mean(1).view(-1)
        # Add zeros to the beginning to align the moving average with episode numbers
        means = torch.cat((torch.zeros(moving_ave - 1), means))
        # Plot the moving average
        plt.plot(means.numpy())

    # Pause for a short time to update the plot (necessary for real-time display)
    plt.pause(0.001)

    # Save the plot to a file if a save path is provided
    if save_path:
        plt.savefig(save_path)

    # Display the plot if not saving only
    if not save_only:
        if is_ipython:
            if not show_result:
                # In non-final result mode, update the plot display
                display.display(plt.gcf())
                #display.clear_output(wait=True)
            else:
                # In final result mode, just display the plot without updating
                display.display(plt.gcf())
        else:
            plt.show()  # For non-IPython environments, display the plot


def plot_episode_by_eps(save_path=None, save_only=True):
    x_values = episode_rewards
    y_values = episode_rewards

    # Normalize x and y values to be between 0 and 1
    x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
    y_normalized = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))

    # Convert list to a PyTorch tensor of type float
    rewards_t = torch.tensor(x_normalized, dtype=torch.float)
    eps_t = torch.tensor(y_normalized, dtype=torch.float)

    plt.figure(figsize=(8, 6))
    plt.plot(rewards_t.numpy(), label='Rewards')
    plt.plot(eps_t.numpy(), label='Epsilon Threshold')

    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.title('(Normalized)')
    plt.legend()

    # Save the plot to a file if a save path is provided
    if save_path:
        plt.savefig(save_path)

    if save_only:
        plt.show()  # For non-IPython environments, display the plot


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 100
    print('GPU')
else:
    num_episodes = 100
    print('Not GPU')

rewards_file = open('./saves2/rewards.txt', 'a')
durations_file = open('./saves2/durations.txt', 'a')
last_save_index = 0
training_start_time = pg.time.get_ticks()

def save_lists_data():
    global last_save_index
    for reward, duration in zip(episode_rewards[last_save_index:], episode_durations[last_save_index:]):
        rewards_file.write(str(reward) + ',')
        durations_file.write(str(duration) + ',')
    rewards_file.flush()
    durations_file.flush()
    last_save_index = len(episode_rewards)

for i_episode in range(load_episode, load_episode + num_episodes):
    # Initialize the environment and get its state
    episode_reward = 0
    episode_start_time = pg.time.get_ticks()
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, done, info = env.step(action.item())
        episode_reward += reward
        reward = torch.tensor([reward], device=device)
        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_finish_time = pg.time.get_ticks()
            episode_duration = episode_finish_time - episode_start_time
            episode_rewards.append(episode_reward)
            episode_durations.append(round(episode_duration/1000, 2))
            print(info)
            if save_model:
                if False and i_episode != 0 and i_episode % 50 == 0:
                    plot_rewards()
                
                if i_episode == 0:
                    nn_saver.save_as_image(f'./saves2/aliens_policy_model_initial.png')
                
                if i_episode != load_episode and i_episode % 10 == 0:
                    if False and i_episode % 50 == 0:
                        plot_rewards(save_only=True, show_result=True, save_path=f'./saves2/aliens_plot_e{i_episode}.png')
                    
                    save_lists_data()
                    
                    torch.save(policy_net, f'./saves2/aliens_policy_model_e{i_episode}_{steps_done}.pth')
                    torch.save(target_net, f'./saves2/aliens_target_model_e{i_episode}_{steps_done}.pth')
            break

print('Complete', 'Step:', steps_done)
save_lists_data()
torch.save(policy_net, f'./saves2/aliens_policy_model_e{load_episode+num_episodes}_{steps_done}.pth')
torch.save(target_net, f'./saves2/aliens_target_model_e{load_episode+num_episodes}_{steps_done}.pth')
nn_saver.save_as_image(f'./saves2/aliens_policy_model_e{load_episode+num_episodes}_{steps_done}.png')
training_finish_time = pg.time.get_ticks()
training_time = training_finish_time - training_start_time
print('Training Duration:', training_time / 1000, 'secs')
#plot_rewards(save_only=True, show_result=True, save_path=f'./saves2/aliens_plot_e{load_episode+num_episodes}.png')
rewards_file.close()
durations_file.close()
env.close()
plt.ioff()
plt.show()