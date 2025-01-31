import gymnasium as gym
import aliens_env
from itertools import count


if __name__ == "__main__":
    env = gym.make('Aliens', render_mode=None, play_sounds=False)
    print('Observations:', env.observation_space.shape[0])
    print('Actions:', env.action_space.n)
    
    seed = None
    for i in count():
        done = False
        observation, info = env.reset(seed=seed)
        env.action_space.seed(seed)
        episode_reward = 0
        
        while not done:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        print(info, 'Reward:', episode_reward)
    env.close()