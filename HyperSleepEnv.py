import random
import gym
from gym import spaces 
import numpy as np

class HyperSleepEnv(gym.Env):
    def __init__(self):
        # Actions our pod can take: increase, decrease or stay same
        self.action_space = spaces.Discrete(3)
        # Temperature array
        self.observation_space = spaces.Box(low=np.array([50]), high=np.array([70]))
        # Set start temp
        self.state = 65 + random.randint(-5,5)
        # Set sleep length
        self.sleep_duration = 60
        # Set reward
        self.reward = 0
        
    def step(self, action):
        self.state += (action - 1)*5 
        # Reduce sleep duration by 1 
        self.sleep_duration -= 1 
        
        # Calculate reward
        if self.state >=60 and self.state <=67: 
            self.reward += 100 
        else: 
            self.reward = -10 
        
        # Check if sleep duration is over
        if self.sleep_duration <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        self.state += random.randint(-5,5)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, self.reward, done, info

    def render(self):
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = np.array([60 + random.randint(-10,10)]).astype(float)
        # Reset shower time
        self.sleep_duration = 60 
        self.reward = 0
        return self.state

