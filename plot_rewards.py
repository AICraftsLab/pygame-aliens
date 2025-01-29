import sys

p10 = 'C:/Users/Sabo/AppData/Local/Programs/Python/Python310/'
p10_scripts = 'C:/Users/Sabo/AppData/Local/Programs/Python/Python310/Scripts/'

#sys.path.append(project_path)
print(sys.path)

#sys.exit()

import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size=100):
    """Compute the moving average of data."""
    # Calculate moving average
    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    # Append zeros to match the length of the original data
    moving_avg = np.concatenate((np.zeros(window_size - 1), moving_avg))
    
    return moving_avg

def visualize_rewards(file_path):
    # Read rewards from the file
    with open(file_path, 'r') as file:
        data = file.read().strip().split(',')
        data = data[:-1]
        rewards = np.array(list(map(float, data)))

    # Calculate moving average
    window_size = 100
    moving_avg = moving_average(rewards, window_size)
    
    # Plot rewards and moving average
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, marker='', linestyle='-', color='b', label='Episode Rewards')
    #plt.plot(np.arange(len(moving_avg)) + window_size // 2, moving_avg, color='r', linestyle='--', label='Moving Average')
    plt.plot(moving_avg, color='r', linestyle='--', label='Moving Average')
    plt.title('Agent Rewards over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.savefig('./saves/rewards_plot.png')
    # Display the plot
    plt.show()


def visualize_rewards_and_durations(rewards_file_path, durations_file_path):
    # Read rewards from the file
    with open(rewards_file_path, 'r') as file:
        data = file.read().strip().split(',')
        data = data[:-1]
        rewards = np.array(list(map(float, data)))

    # Read rewards from the file
    with open(durations_file_path, 'r') as file:
        data = file.read().strip().split(',')
        data = data[:-1]
        durations = np.array(list(map(float, data)))

    # Calculate moving average
    window_size = 100
    rewards_moving_avg = moving_average(rewards, window_size)
    durations_moving_avg = moving_average(durations, window_size)
    
    # Plot rewards and moving average
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, linestyle='-', color='b', label='Episode Rewards')
    #plt.plot(np.arange(len(moving_avg)) + window_size // 2, moving_avg, color='r', linestyle='--', label='Moving Average')
    plt.plot(rewards_moving_avg, color='r', linestyle='--', label='Rewards MA')
    
    plt.plot(durations, label='Episode Durations')
    #plt.plot(np.arange(len(moving_avg)) + window_size // 2, moving_avg, color='r', linestyle='--', label='Moving Average')
    plt.plot(durations_moving_avg, linestyle=':', label='Durations MA')
    
    
    plt.title('Agent Rewards over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Reward/Durations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.savefig('./saves/rewards_plot.png')
    # Display the plot
    plt.show()
    
rewards_file_path = './saves/rewards.txt'
durations_file_path = './saves/durations.txt'
visualize_rewards(rewards_file_path)
#visualize_rewards_and_durations(rewards_file_path, durations_file_path)