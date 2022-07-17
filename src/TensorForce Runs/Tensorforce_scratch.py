from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
import numpy as np
import time

max_episode_timesteps = 1000

max_timesteps = 1000

action_decode = np.zeros((81, 4))
count = 0
vals = range(-1, 2)  # gets -1, 0, and 1 (2 is excluded in python)
for i in vals:
    for j in vals:
        for k in vals:
            for l in vals:
                action_decode[count] = [i, j, k, l]
                count = count + 1


Q_network = [
    dict(type='dense', size=170, activation='tanh'),
    dict(type='dense', size=170, activation='tanh')
]

environment = Environment.create(
    environment='gym', level='BipedalWalker-v3', max_episode_timesteps=max_episode_timesteps
)

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

alpha = 1
epsilon = 0.1
render_freq = 10
print_freq = 1
# for episode in range(100):
#     st = time.time()
#     # Initialize episode
#     states = environment.reset()
#     terminal = False
#     epsilon = alpha * epsilon
#
#     if episode % print_freq == 0:
#         print(episode)
#     while not terminal:
#         # Episode timestep
#         if episode % render_freq == 0:
#             environment.environment.render()
#
#         actions = action_decode[agent.act(states=states)]
#
#         if np.random.random_sample() < epsilon:
#             actions = action_decode[np.random.randint(0, 81)]
#
#         states, terminal, reward = environment.execute(actions=actions)
#         agent.observe(terminal=terminal, reward=reward)
#
#     en = time.time()
#     print(en - st)

runner = Runner(
    agent=agent,
    environment=environment,
    max_episode_timesteps=max_episode_timesteps
)

runner.run(num_episodes=200)

runner.run(num_episodes=100, evaluation=True)

runner.close()

agent.close()
environment.close()
