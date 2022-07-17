import torch
import gym
import numpy as np
import random
import copy
import time


def disc_actions(action_space, bins):
    """
    Given action space and number of bins per action variable,
    return list of discretized action space.
    """
    low_bound = action_space.low[0]  # lower bound of action space is the same for the 4 action variables
    high_bound = action_space.high[0]  # upper bound of action space is the same for the 4 action variables
    a_space_bins = np.round(np.linspace(low_bound, high_bound, num=bins), 2)  # list of evenly spaced actions

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

alpha = 0.001    # learning rate
epsilon = 1      # greedy policy percentage
bins = 3         # bins per action variable
gamma = 0.97     # discount factor

a_space_disc = disc_actions(action_space, bins)  # return discretized action space

n_state = env.observation_space.shape[0]  # 24 state variables
n_action = len(a_space_disc)  # n_action possible discretized actions  = bins ** 4

# initialize neural network
Q = torch.nn.Sequential(
    torch.nn.Linear(n_state, 55),
    torch.nn.ReLU(),
    torch.nn.Linear(55, 55),
    torch.nn.ReLU(),
    torch.nn.Linear(55, 55),
    torch.nn.ReLU(),
    torch.nn.Linear(55, n_action)
    ).double()

optimizer = torch.optim.Adam(Q.parameters(), lr=alpha)


batches = 1000    # iterations of Q_ update
episodes_per_batch = 10    # iterations per Q_ update

print_freq = 50
render_freq = 90000
data = []
episode = 0
for batch in range(batches):
    epsilon = 1 - (batch / (batches - 1))
    Q_ = copy.deepcopy(Q)
    for episode_in_batch in range(episodes_per_batch):
        observation = torch.as_tensor(env.reset()).double()
        reward_sum = 0.0
        num_steps = 0
        done = False
        render = False
        if episode % render_freq == 0:
            #render = True
            pass
        while not done:
            if render:
                env.render()
            action = e_greedy_step(Q_, observation, epsilon)
            observation_next, reward, done, info = env.step(action)
            reward = max(min(reward, 1), -1)
            reward_sum = reward_sum + reward
            num_steps = num_steps + 1

            Q_target = reward + (1 - done) * gamma * act_optimal(Q_, observation_next)[1]  # estimated optimal u
            Q_output = Q(observation)[a_space_disc.index(action)]  # u given by Q(s, a) received when taking action a from s
            observation = torch.as_tensor(observation_next).double()
            criterion = torch.nn.MSELoss()  # squared error loss function
            loss = criterion(Q_output, Q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        episode = episode + 1
        data.append([episode, batch + 1, episode_in_batch + 1, reward_sum, num_steps, epsilon])
        if episode % print_freq == 0:
            print('Episode: ', episode, '   Batch: ', batch + 1, '   Reward: ', reward_sum, '   Steps: ', num_steps)

# env.close()
end = time.time()
runtime = end - start
data.insert(0, [runtime, alpha, 0, 0, 0, 0])
print("Runtime: ", str(runtime))

# torch.save(Q, "Q_2layers_3bins.pt")
np.savetxt("Bipedal_Data_DQN.csv", data, delimiter=", ", fmt='% s')

