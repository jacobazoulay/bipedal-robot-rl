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

# agent = Agent.create(
#     agent='dqn',
#     states=environment.states(),
#     actions=dict(type="int", shape=(), num_values=81),
#     max_episode_timesteps=max_timesteps,
#     memory=10000,
#     batch_size=16,
#     learning_rate=1e-2,
#     network=Q_network,
#     discount=0.95,
#     horizon=100
# )

agent = Agent.load(directory='model-numpy', format='numpy', environment=environment, filename='agent-2000')

# Evaluate for 100 episodes
st = time.time()
sum_rewards = 0.0
for _ in range(10):
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        environment.environment.render()
        actions, internals = agent.act(
            states=states, internals=internals, independent=True, deterministic=True
        )
        actions = action_decode[actions]
        states, terminal, reward = environment.execute(actions=actions)
        sum_rewards += reward
print('Mean evaluation return:', sum_rewards / 100.0)

en = time.time()
print(en - st)

agent.close()
environment.close()