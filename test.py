import gym
import os 
import wandb
from HyperSleepEnv import HyperSleepEnv
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "HyperSleep",
}
run = wandb.init(
    project="intro_to_gym",
    config=config,
    sync_tensorboard=True,  
    monitor_gym=True,  
    save_code=True,  
)

env = HyperSleepEnv()
env = Monitor(env)
env = DummyVecEnv([lambda:env])

model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
total_timesteps=config["total_timesteps"],
callback=WandbCallback(
gradient_save_freq=10,
model_save_path=f"models/{run.id}",
verbose=2,
),)

PPO_path = os.path.join('Training', 'Saved Models', 'PPO_HyperSleep_25k')
model.save(PPO_path)

evaluate_policy(model, env, n_eval_episodes=10)
run.finish()
