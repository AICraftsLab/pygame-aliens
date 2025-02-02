import os
import numpy as np
import gymnasium as gym
import aliens_env
from itertools import count

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy


if __name__ == "__main__":
    env_id = 'Aliens'
    save_dir = 'run1'
    seed = None
    
    model_file_path = 'run1/model_15000000_steps.zip'
    stats_file_path = 'run1/model_vecnormalize_15000000_steps.pkl'
    
    env = [lambda: gym.make(env_id,
                            render_mode='human',
                            play_sounds=True)]
                            
    vec_env = DummyVecEnv(env)
    vec_env.seed(seed)
    env = vec_env.envs[0]
    
    vec_env = VecNormalize.load(stats_file_path, vec_env)
    vec_env.training = False  # Do not update stats at test time
    #vec_env.norm_reward = False  # Not needed at test time
    
    model = PPO.load(model_file_path)
    #print(model.policy)
    #print()
    print('Model:', model_file_path)
    
    for i in count():
        obs, info = env.reset()
        episode_reward = 0
        done = False
        while not done:
            obs = vec_env.normalize_obs(obs)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        print(info, 'Reward:', round(episode_reward, 2))

