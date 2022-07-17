from tensorforce.agents import Agent
from tensorforce.environments import Environment
import numpy as np
import time

# definitely not the best way to do this but I don't care
action_decode = np.zeros((81, 4))
count = 0
vals = range(-1, 2)  # gets -1, 0, and 1 (2 is excluded in python)
for i in vals:
    for j in vals:
        for k in vals:
            for l in vals:
                action_decode[count] = [i, j, k, l]
                count = count + 1

max_timesteps = 1000

environment = Environment.create(
    environment='gym', level='BipedalWalker-v3', max_episode_timesteps=max_timesteps
)

Q_network = [
    dict(type='dense', size=170, activation='tanh'),
    dict(type='dense', size=170, activation='tanh')
]

alpha = 1  # 0.9999
epsilon = 0.2

agent = Agent.create(
    agent='dqn',
    states=environment.states(),
    actions=dict(type="int", shape=(), num_values=81),
    max_episode_timesteps=max_timesteps,
    memory=10000,
    batch_size=16,
    learning_rate=1e-3,
    network=Q_network,
    discount=0.99,
    horizon=100
)


print_freq = 25
st = time.time()
reward_totals = []
episodes = 2000
for episode in range(episodes):
    # Initialize episode
    states = environment.reset()
    terminal = False
    sum_rewards = 0.0
    num_updates = 0
    epsilon = 1 - (episode/(episodes - 1))

    while not terminal:
        actions = action_decode[agent.act(states=states)]
        if np.random.random_sample() < epsilon:
            actions = action_decode[np.random.randint(0, 81)]
        states, terminal, reward = environment.execute(actions=actions)
        num_updates += agent.observe(terminal=terminal, reward=reward)
        sum_rewards += reward

    reward_totals.append([episode, sum_rewards, num_updates])
    if episode % print_freq == 0:
        print('Episode {}: return={} updates={}'.format(episode, sum_rewards, num_updates))

agent.save(directory='model-numpy', format='numpy', append='episodes')
print(reward_totals)

en = time.time()
print(en - st)

agent.close()
environment.close()