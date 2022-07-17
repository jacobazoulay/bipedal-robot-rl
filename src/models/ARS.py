import numpy as np
import gym
import time

episodes = 1500                 # number of training episodes
learning_rate = 0.02            # learning rate
num_deltas = 16                 # number of random nearby points
num_best_deltas = 16            # highest reward points used to update model
noise = 0.03                    # random noise
env_name = 'BipedalWalker-v3'   # environment
render_freq = 1                 # render frequency

env = gym.make(env_name)
episode_length = env.spec.max_episode_steps     # 1600 max steps per episode until done
input_size = env.observation_space.shape[0]     # 24 observation variables
output_size = env.action_space.shape[0]         # 4 action variables

n = np.zeros(input_size)                        # list of 24 numbers, counts of observed states
mean = np.zeros(input_size)                     # mean of observed state
mean_dif = np.zeros(input_size)                 # mean dif of observed states used to calculated std
std = np.zeros(input_size)                      # std of observed states

theta = np.zeros((output_size, input_size))     # weights of network 4x24 matrix


def normalize(observation):
    """
    Given raw state observation from environment, return normalized state observation.
    """
    global n
    global mean
    global mean_dif
    global std
    n = n + 1.0                                                                # update observation count
    last_mean = mean.copy()                                                    # save previous mean
    mean = mean + (observation - mean) / n                                     # update running mean
    mean_dif = mean_dif + (observation - last_mean) * (observation - mean)     # mean dif used to calculate std
    std = np.sqrt((mean_dif / n).clip(min=0.01))                               # std calculation
    norm_val = (observation - mean) / std                                      # observation normalization
    return norm_val


def act(observation, delta=None, direction=None):
    """
    Given normalized state observation, return action given by network.
    Optional parameters to add or subtract variance to network weights when calculating action.
    """
    if direction is None:
        return theta.dot(observation)
    elif direction == '+':
        return (theta + noise * delta).dot(observation)
    elif direction == '-':
        return (theta - noise * delta).dot(observation)


def update(rollouts, sigma_rewards):
    """
    Given rollouts in order of max reward (3-tuple of pos_r, neg_r, and delta) and std of rewards, update theta.
    """
    global theta
    step = np.zeros(theta.shape)
    for r_pos, r_neg, delta in rollouts:
        step = step + (r_pos -r_neg) * delta
    theta = theta + learning_rate / (num_best_deltas * sigma_rewards) * step


def explore(direction=None, delta=None, render=False):
    """
    Run through one episode using current neural network weights and normalizing the observations.
    Optional direction and delta parameters passed to act() function to return action.
    Optional render parameter determines if animation is rendered.
    Returns total reward from episode (normalized so each time step reward is between -1 and 1).
    """
    state = env.reset()
    done = False
    num_steps = 0.0
    reward_sum = 0.0
    while not done and num_steps < episode_length:
        if render:
            env.render()
        state = normalize(state)
        action = act(state, delta, direction)
        state, reward, done, info = env.step(action)
        reward = max(min(reward, 1), -1)
        reward_sum = reward_sum + reward
        num_steps = num_steps + 1
    return reward_sum, num_steps


total_start = time.time()
for learn_rate in [0.02, 0.06, 0.001]:
    learning_rate = learn_rate
    start = time.time()
    data = []
    n = np.zeros(input_size)  # list of 24 numbers, counts of observed states
    mean = np.zeros(input_size)  # mean of observed state
    mean_dif = np.zeros(input_size)  # mean dif of observed states used to calculated std
    std = np.zeros(input_size)  # std of observed states

    theta = np.zeros((output_size, input_size))  # weights of network 4x24 matrix

    for episode in range(episodes):
        pos_rewards = [0] * num_deltas    # list of rewards from positive deltas
        neg_rewards = [0] * num_deltas    # list of rewards from negative deltas
        deltas = [np.random.randn(*theta.shape) for delta in range(num_deltas)]  # random noise to apply to weights
        render = False
        avg_steps = 0.0
        if episode % render_freq == 0 and episode != 0:
            render = True

        for i in range(num_deltas):
            pos_rewards[i], num_step_pos = explore(direction="+", delta=deltas[i], render=render)
            neg_rewards[i], num_step_neg = explore(direction="-", delta=deltas[i], render=render)
            avg_steps = avg_steps + num_step_pos + num_step_neg

        avg_steps = avg_steps / (2 * num_deltas)
        sigma_rewards = np.array(pos_rewards + neg_rewards).std()                                           # std of rewards of batch of rollouts with random noise in weights
        scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(pos_rewards, neg_rewards))}   # create dictionary (length of num_deltas) with highest rewards
        order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:num_best_deltas]              # order of rollouts sorted by max reward
        rollouts = [(pos_rewards[k], neg_rewards[k], deltas[k]) for k in order]                             # rollouts in order of max reward

        update(rollouts, sigma_rewards)

        reward_eval, num_steps = explore(render=render)
        data_point = [episode + 1, reward_eval, num_steps, avg_steps, sigma_rewards]
        data.append(data_point)
        if episode % 100 == 0:
            print('Episode: ', episode + 1, '   Reward: ', reward_eval, '   Alpha: ', learning_rate)
    print(theta)
    end = time.time()
    runtime = end - start
    data.insert(0, [runtime, learning_rate, 0, 0, 0])
    filename = "Bipedal_data_" + str(int(learning_rate * 1000)) + ".csv"
    np.savetxt(filename,
               data,
               delimiter=", ",
               fmt='% s')

total_end = time.time()
print(total_end - total_start)