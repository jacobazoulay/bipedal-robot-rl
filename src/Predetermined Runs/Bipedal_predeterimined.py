import gym
import time
import numpy as np

env = gym.make('BipedalWalker-v3')
start = time.time()
render_freq = 1
render = True
episodes = 1000


def policy_walk():
    """
    Deterministically inch forward and return collected data.
    """
    data = []
    for episode in range(episodes):
        reward_sum = 0.0
        num_step = 0
        done = False
        env.reset()
        for i in range(25):
            if not done:
                if render:
                    env.render()
                action = [1, 0, 0, 0]
                observation_next, reward, done, info = env.step(action)
                reward_sum = reward_sum + max(min(reward, 1), -1)
                num_step = num_step + 1

        for i in range(25):
            if not done:
                if render:
                    env.render()
                action = [0, 0, 0, 0]
                observation_next, reward, done, info = env.step(action)
                reward_sum = reward_sum + max(min(reward, 1), -1)
                num_step = num_step + 1

        for i in range(90):
            if not done:
                if render:
                    env.render()
                action = [1, 1, 0, 0.4]
                observation_next, reward, done, info = env.step(action)
                reward_sum = reward_sum + max(min(reward, 1), -1)
                num_step = num_step + 1

        for i in range(40):
            if not done:
                if render:
                    env.render()
                action = [0.6, -1, 0, 0.4]
                observation_next, reward, done, info = env.step(action)
                reward_sum = reward_sum + max(min(reward, 1), -1)
                num_step = num_step + 1

        while not done:
            for i in range(40):
                if not done:
                    if render:
                        env.render()
                    action = [1, 1, -0.3, 0.6]
                    observation_next, reward, done, info = env.step(action)
                    reward_sum = reward_sum + max(min(reward, 1), -1)
                    num_step = num_step + 1

            for i in range(40):
                if not done:
                    if render:
                        env.render()
                    action = [0.6, -0.4, 0, 0.4]
                    observation_next, reward, done, info = env.step(action)
                    reward_sum = reward_sum + max(min(reward, 1), -1)
                    num_step = num_step + 1

        data.append([episode + 1, reward_sum, num_step])
        if episode % render_freq == 0:
            print('Episode: ', episode + 1, '  Reward: ', reward_sum, '  Steps: ', num_step)
    return data


def policy_stay_still():
    """
    Deterministically stay still and return data.
    """
    data = []
    for episode in range(episodes):
        reward_sum = 0.0
        num_step = 0
        done = False
        env.reset()
        for i in range(3):
            if not done:
                if render:
                    env.render()
                action = [1, 0, -1, 0]
                observation_next, reward, done, info = env.step(action)
                reward_sum = reward_sum + max(min(reward, 1), -1)
                num_step = num_step + 1

        while not done:
            if render:
                env.render()
            action = [0, 0, 0, 0]
            observation_next, reward, done, info = env.step(action)
            reward_sum = reward_sum + max(min(reward, 1), -1)
            num_step = num_step + 1

        data.append([episode + 1, reward_sum, num_step])
        if episode % render_freq == 0:
            print('Episode: ', episode + 1, '  Reward: ', reward_sum, '  Steps: ', num_step)
    return data


def policy_random():
    data = []
    for episode in range(episodes):
        reward_sum = 0.0
        num_step = 0
        done = False
        env.reset()
        while not done:
            if render:
                env.render()
            action = env.action_space.sample()
            observation_next, reward, done, info = env.step(action)
            reward_sum = reward_sum + max(min(reward, 1), -1)
            num_step = num_step + 1

        data.append([episode + 1, reward_sum, num_step])
        if episode % render_freq == 0:
            print('Episode: ', episode + 1, '  Reward: ', reward_sum, '  Steps: ', num_step)
    return data


data = policy_random()
print(data)
end = time.time()
runtime = end - start
data.insert(0, [runtime, 0, 0])
np.savetxt('Rewards_walk_forward.csv',
           data,
           delimiter=", ",
           fmt='% s')
print(runtime)
env.close()
