import os
import numpy as np
import gymnasium as gym
import aliens_env
from itertools import count

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Save normalization stats callback
class SaveNormStatsCallback(BaseCallback):
    def __init__(self, vec_env: VecNormalize, filename: str = 'vecnormalize.pkl', save_dir: str = './', verbose: int = 1):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.save_dir = save_dir
        self.filename = filename

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        save_path = os.path.join(self.save_dir, self.filename)
        self.vec_env.save(save_path)
        if self.verbose >= 1:
            print(f"Saving best model normalizing stats to {save_path}")

        return True


if __name__ == "__main__":
    env_id = 'Aliens'
    save_dir = 'run1'
    seed = None
    tensorboard_log = 'tensorboard_logs'
    
    # For resuming training
    is_new_training = True
    model_file_path = None
    stats_file_path = None
    
    # Save_dir should not exist if is_new_training
    os.makedirs(save_dir, exist_ok= not is_new_training)
    
    timesteps = 5e6
    num_cpu = 20  # Env nums
    vec_env_cls = DummyVecEnv
    
    vec_env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=vec_env_cls, seed=seed)
    
    if is_new_training:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    else:
        vec_env = VecNormalize.load(stats_file_path, vec_env)
    
    checkpoint_callback = CheckpointCallback(
      save_freq=10000,
      save_path=save_dir,
      name_prefix="model",
      save_replay_buffer=False,
      save_vecnormalize=True,
      verbose=1,
    )
    
    if is_new_training:
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=tensorboard_log)
    else:
        model = PPO.load(model_file_path, vec_env, verbose=1, tensorboard_log=tensorboard_log)
        print('Loaded model:', model_file_path)
    
    reset_num_timesteps = is_new_training
    model.learn(total_timesteps=int(timesteps), callback=checkpoint_callback, reset_num_timesteps=reset_num_timesteps, tb_log_name=save_dir)
    
    print('Training complete')
    print('Name:', save_dir)
