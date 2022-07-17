import torch
import gym
import numpy as np
import random
import copy
import time
import os
import csv


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


def e_greedy_step(Q, observation, epsilon):
    """
    Given Q neural network, state observations (list of 24 state variables), and epsilon
    return random policy with epsilon probability. Otherwise act optimally.
    """
    if np.random.random_sample() < epsilon:
        return random.choice(a_space_disc)
    else:
        return act_optimal(Q, observation)[0]


start = time.time()

env = gym.make('BipedalWalker-v3')

obs_space = env.observation_space  # observation space
action_space = env.action_space    # action space

observation = torch.as_tensor(env.reset()).double()  # initial observation

alpha = 0.001    # learning rate
epsilon = 1      # greedy policy percentage
bins = 3         # bins per action variable
gamma = 0.95     # discount factor

a_space_disc = disc_actions(action_space, bins)  # return discretized action space

n_state = env.observation_space.shape[0]  # 24 state variables
n_action = len(a_space_disc)  # n_action possible discretized actions  = bins ** 4

# initialize neural network
Q = torch.nn.Sequential(
    torch.nn.Linear(n_state, 80),
    torch.nn.ReLU(),
    torch.nn.Linear(80, 180),
    torch.nn.ReLU(),
    torch.nn.Linear(180, 180),
    torch.nn.ReLU(),
    torch.nn.Linear(180, n_action)
    ).double()

optimizer = torch.optim.Adam(Q.parameters(), lr=alpha)


k_1 = 1800   # iterations of Q_ update
k_2 = 1000   # iterations per Q_ update

reward_sums = []
reward_sum = 0
done = False
for i in range(k_1):
    if i % 1 == 0:
        epsilon = 1 - (i / (k_1 - 1))
        #print(epsilon)
    Q_ = copy.deepcopy(Q)
    for j in range(k_2):
        #env.render()
        if done:
            reward_sums.append(reward_sum)
            observation = torch.as_tensor(env.reset()).double()
            reward_sum = 0
        action = e_greedy_step(Q_, observation, epsilon)
        observation_next, reward, done, info = env.step(action)
        reward_sum = reward_sum + reward

        Q_target = reward + (1 - done) * gamma * act_optimal(Q_, observation_next)[1]  # estimated optimal u
        Q_output = Q(observation)[a_space_disc.index(action)]  # u given by Q(s, a) received when taking action a from s
        observation = torch.as_tensor(observation_next).double()
        #print(action, ", ", reward)
        criterion = torch.nn.MSELoss()  # squared error loss function
        loss = criterion(Q_output, Q_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#env.close()

print(reward_sums)
end = time.time()
print("Runtime: ", str(end - start))

torch.save(Q, "Q_2layers_3bins.pt")
np.savetxt("Rewards.csv", reward_sums, delimiter=", ", fmt='% s')

