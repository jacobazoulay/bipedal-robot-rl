import torch
import gym
import numpy as np
import random
import copy
import time
import os


def disc_actions(action_space, bins):
    """
    Given action space and number of bins per action variable,
    return list of discretized action space.
    """
    low_bound = action_space.low[0]  # lower bound of action space is the same for the 4 action variables
    high_bound = action_space.high[0]  # upper bound of action space is the same for the 4 action variables
    a_space_bins = np.round(np.linspace(low_bound, high_bound, num=bins + 2)[1:bins + 1], 2)  # list of evenly spaced actions

    a_space_disc = []
    for i in a_space_bins:
        for j in a_space_bins:
            for k in a_space_bins:
                for m in a_space_bins:
                    a_space_disc.append([i, j, k, m])
    return a_space_disc


def act_optimal(Q, observation):
    """
    Given Q neural network and state observations (list of 24 state variables),
    return best action and action-value determined by Q.
    """
    observation = torch.as_tensor(observation).double()
    Q_out = Q(observation)
    action_u = torch.max(Q_out)
    action_idx = torch.argmax(Q_out)
    action = a_space_disc[action_idx]
    return action, action_u


def explore_step(Q, observation, epsilon):
    """
    Given Q neural network, state observations (list of 24 state variables), and epsilon
    return random policy with epsilon probability. Otherwise act optimally.
    """
    if np.random.random_sample() < epsilon:
        return random.choice(a_space_disc)
    else:
        return act_optimal(Q, observation)[0]


env = gym.make('BipedalWalker-v3')

obs_space = env.observation_space
action_space = env.action_space

observation = torch.as_tensor(env.reset()).double()

Q = torch.load("Q_2layers_3bins.pt")
Q.eval()

bins = int(len(Q(observation)) ** (1/4))  # bins per action variable
a_space_disc = disc_actions(action_space, bins)


done = False
for i in range(2000):
    env.render()
    if done:
        observation = torch.as_tensor(env.reset()).double()
    action = act_optimal(Q, observation)[0]
    observation, reward, done, info = env.step(action)
    observation = torch.as_tensor(observation).double()
    print(observation, action, ", ", reward)

env.close()



