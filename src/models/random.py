import gym
import random
import numpy as np


env = gym.make("BipedalWalker-v3")


def random_games():
    # Each of this episode is its own game.
    action_size = env.action_space.shape[0]
    for episode in range(10):
        env.reset()
        count = 0
        # this is each frame, up to 1600...but we wont make it that far with random.
        while True:
            count = count + 1
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be any of one how in list on 4, for example [0 1 0 0]
            action = np.random.uniform(-1.0, 1.0, size=action_size)

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            next_state, reward, done, info = env.step(action)

            # lets print everything in one line:
            if count % 100 == 0:
                print(reward, action)
            if done:
                print(count)
                break


def random_games_2():
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
            if t == 99:
                print("Episode finished after {} timesteps".format(t + 1))
    env.close()


def test():
    # Observation and action space
    obs_space = env.observation_space
    action_space = env.action_space
    print(obs_space.shape[0])  # 24 state variables
    print("The observation space: {}".format(obs_space), '\n')
    print("The action space: {}".format(action_space), '\n')

    observation = env.reset()
    print(observation, '\n')
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    print(action, '\n')
    print(observation, '\n')
    print(reward)
    print(done)

    env.close()