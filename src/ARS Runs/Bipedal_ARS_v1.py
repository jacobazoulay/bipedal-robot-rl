# importing libraries
import os
import numpy as np
import gym
import pybullet_envs
from gym import wrappers

ENV_NAME = 'BipedalWalker-v3'
# ENV_NAME = 'HalfCheetahBulletEnv-v0'


# setting hyper parameters
class Hp():
    # Hyperparameter
    # Store hyperparameter for turning
    def __init__(self,
                 nb_steps=1000,
                 episode_length=2000,
                 learning_rate=0.02,
                 num_deltas=16,
                 num_best_deltas=16,
                 noise=0.03,
                 seed=1,
                 env_name='BipedalWalker-v3',
                 record_every=50):
        '''
        nb_steps = how many fully training steps we will run

        episode_length = maximun # of steps in episode

        learning_rate = how much we update the weights in each iterations

        num_delta = # of variations of random noise we generate in each tarining steps
                    each one of this we be run for 2 episodes for +ve and -ve versions

        num_best_delta = # of deltas we use to update our weights sorted by the highest
                        reward from both positive and negative variations. this should be
                        less than or equal to num_deltas

        noise = strength of random noise

        seed = random seed use for generating the noise

        env_name = name of AI gym environment

        record_every = record new video after this many # of steps has passed
        '''
        # defining variables of objects
        self.nb_steps = nb_steps  # no. of training loops i.e. no. of times we update the weights
        self.episode_length = episode_length  # max. time AI walk on field
        self.learning_rate = learning_rate  # how fast AI is learning
        self.num_deltas = num_deltas  # no. of directions
        self.num_best_deltas = num_best_deltas  # no.of best directions. keep same as nb_directions in starting
        assert self.num_best_deltas <= self.num_deltas
        self.noise = noise  # Noise introduced
        self.seed = seed
        self.env_name = env_name  # name of environment
        self.record_every = record_every


# Normalizing the states (to improve performance)
class Normalizer():  # refer page 7 section 3.2 of paper
    # Normalize the input
    # init fn creates empty array which is size of our input space
    def __init__(self, nb_inputs):  # nb_inputs is no. of perceptrons
        # initializing the variables required for normalization
        self.n = np.zeros(nb_inputs)  # total number of states. vector of zeros equal to no. of perceptrons
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    # compute running average as well as variance
    def observe(self, x):  # x is new state. function observe is called everytime we observe a new state
        self.n += 1.0  # to make it float
        last_mean = self.mean.copy()  # saving mean before updating it
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)

        # clip(min = 1e-2) is used to make sure self.var is never equal to zero
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)  # std. deviation
        return (inputs - obs_mean) / obs_std  # normalized values


# Building the AI
# Algorithm is based on exploration on the space of policies. We will explore
# many policies and converge on the one which returns the best output
class Policy():
    # Responsible to generate random noise and update policy from rollout and
    # turning inputs into actions
    def __init__(self, input_size, output_size, hp):  # there are many outputs
        self.theta = np.zeros((output_size, input_size))  # matrix of weights of neurons of perceptron
        self.hp = hp

    # Evaluate functions turns input into actions
    # page 6 algorithm 2 step5 V2. delta - matrix of small number helps in choosing direction.
    # direction can have 3 values: +ve, -ve and none.
    def evaluate(self, input, delta=None, direction=None):
        if direction is None:
            return self.theta.dot(input)  # matrix multiplication theta x input
        elif direction == '+':
            return (self.theta + self.hp.noise * delta).dot(input)  # hp is object of class Hp where noise is defined
        elif direction == '-':
            return (self.theta - self.hp.noise * delta).dot(input)

    def sample_deltas(self):
        # returning matrix(of same size of theta matrix) of random small values of delta.
        # *self.theta.shape gives the dimension of theta matrix
        return [np.random.randn(*self.theta.shape) for _ in range(self.hp.num_deltas)]

    # step 7. rollouts contains rewards in positive and negative direction, and delta
    def update(self, rollouts, sigma_rewards):
        # sigma_rewards is standard deviation of rewards
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta
        self.theta += self.hp.learning_rate / (self.hp.num_best_deltas * sigma_rewards) * step


class ArsTrainer():
    def __init__(self,
                 hp=None,
                 input_size=None,
                 output_size=None,
                 normalizer=None,
                 policy=None,
                 monitor_dir=None):

        self.hp = hp or Hp()
        np.random.seed(self.hp.seed)
        self.env = gym.make(self.hp.env_name)
        self.hp.episode_length = self.env.spec.max_episode_steps or self.hp.episode_length
        self.input_size = input_size or self.env.observation_space.shape[0]
        self.output_size = output_size or self.env.action_space.shape[0]
        self.normalizer = normalizer or Normalizer(self.input_size)
        self.policy = policy or Policy(self.input_size, self.output_size, self.hp)
        # self.record_video = False

    # Explore the policy on one specific direction and over one episode
    def explore(self, direction=None, delta=None, render=False):
        state = self.env.reset()  # returns first state. env is object of pybullet
        done = False  # boolean. true if end of episode. False in starting bcoz episode is not done
        num_plays = 0.0  # no. of actions
        sum_rewards = 0.0  # accumulative rewards
        while not done and num_plays < self.hp.episode_length:
            if render:
                self.env.render()
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state)  # normalizing state
            action = self.policy.evaluate(state, delta, direction)  # evaluating the policy

            # step fn from environment object of pybullet returns next state of
            # the environment the reward obtained after playing the action and whether or not the episode is done
            state, reward, done, _ = self.env.step(action)
            # +1 for very high positive reward and -1 for very high negative reward, to avoid bais
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_plays += 1
        return sum_rewards

    # Training the AI
    def train(self):  # step 3
        for step in range(self.hp.nb_steps):  # hp.nb_steps refer to no. of training loops
            # initialize the random noise deltas and the positive/negative rewards
            deltas = self.policy.sample_deltas()  # we get 16 matrixs for one for each 16 directions
            positive_rewards = [0] * self.hp.num_deltas  # list of 16 zeros
            # negative rewards does not mean less than zero. it means rewards in opposite direction
            negative_rewards = [0] * self.hp.num_deltas
            render = False
            if step % 200 == 0:
                render = True

            # play an episode each with positive deltas and negative deltas, collect rewards
            # getting the positive rewards in positive direction
            for k in range(self.hp.num_deltas):  # looping through the no. of directions
                # direction == positive as in line 55. deltas[k] refers to delta in kth direction
                positive_rewards[k] = self.explore(direction="+", delta=deltas[k], render=render)
                # direction == positive as in line 57. deltas[k] refers to delta in kth direction
                negative_rewards[k] = self.explore(direction="-", delta=deltas[k], render=render)

            # Gethering all the positive and negative rewards to compute the
            # standard deviation of these rewards. section 3.1
            # Compute the standard deviation of all rewards
            sigma_rewards = np.array(positive_rewards + negative_rewards).std()

            # Sort the rollouts by the max(r_pos, r_neg) and select the deltas with best rewards
            # storing in a dictionary
            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            # sorting by keys. we are only considering the best directions.
            order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.hp.num_best_deltas]
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

            # Update the policy
            # Updating our policy. Step 7. weights are updated here to reach highest reward
            self.policy.update(rollouts, sigma_rewards)

            # Play an episode with the new weights and print the score
            # printing the final reward of the policy after update
            reward_evaluation = self.explore(render=render)  # direction and delta are none by default
            print('Step: ', step, 'Reward: ', reward_evaluation)


# Main code
if __name__ == '__main__':
    hp = Hp(seed=1946, env_name=ENV_NAME)
    trainer = ArsTrainer(hp=hp)
    trainer.train()