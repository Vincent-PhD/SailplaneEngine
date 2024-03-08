import gym
import numpy as np
from Engines.wing_performance_engine import *


class CustomEnv(gym.Env):
    def __init__(self):
        # Actions our RL agent can take to modify the airfoil shape at a given station
        self.action_space = gym.spaces.Box(
            low=np.array(
                [
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                ]
            ),
            high=np.array(
                [
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                ]
            ),
            dtype=np.float32,
        )
        # Define the plausible lift to drag ration observation space
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([100]))
        # Define the starting state: the lift to drag ratio of the sailplane for the randomly
        # sampled action space and the first operating condtion
        inital_performance_calc = get_wing_performance(X=self.observation_space, V=230, Wing_Planform_Dat_File)[0]
        self.state = inital_performance_calc
        # Set flight duration
        self.flight_duration = 0
        # Set initial flight speed
        self.flight_speed = 230-10
        # Set reward
        self.reward = 0

    def step(self, action):
        # Calculate performance
        step_performance_calc = get_wing_performance(X=self.observation_space, V=self.flight_speed, Wing_Planform_Dat_File)
        # Set new state
        self.state += step_performance_calc[0]
        # Update flight speed
        self.flight_speed -=10
        # Update flight duration
        self.flight_duration+=1
        # Update reward
        self.reward += step_performance_calc[1]
        # Set placeholder for info
        info = {}

        # Check if flight duration is over
        if self.flight_duration == 13: 
            done = True
        else:
            done = False

        
        # Return step information
        return self.state, self.reward, done, info


    def render(self):
        pass

    def reset(self):
        pass
