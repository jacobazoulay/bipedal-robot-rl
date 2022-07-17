# Teaching a Robot to Walk Using Reinforcement Learning

This project applies Deep Q-Learning (DQN) and Augmented Random Search (ARS) to teach a simulated two-dimensional bipedal robot how to walk using the OpenAI Gym BipedalWalker-v3 environment. Deep Q-learning did not yield a high reward policy, often prematurely converging to suboptimal local maxima likely due to the coarsely discretized action space. ARS, however, resulted in a better trained robot, and produced an optimal policy which officially "solves" the BipedalWalker-v3 problem. Various naive policies, including a random policy, a manually encoded inch forward policy, and a stay still policy, were used as benchmarks to evaluate the proficiency of the learning algorithm results.

This is the final project for AA228 Decision Making Under Uncertainty course at Stanford University.

The final paper can be found [here](https://arxiv.org/abs/2112.07031).


## Directory Structure

- **assets**: contains images for documentation and stored dqn model parameters
- **data**: contains results and performance metrics for various tested models.
- **src**:
	- eval: contains model evaluation functions
	- models: contains various robot agent models
		- [dqn](https://github.com/jacobazoulay/bipedal-robot-rl/tree/main/src/models/dqn "dqn"): directory containing PyTorch and TensorForce versions of Deep Q-Learning models and training functions.
		- [ARS.py](https://github.com/jacobazoulay/bipedal-robot-rl/blob/main/src/models/ARS.py "ARS.py"): Augmented Random Search (ARS) model and training function.
		- [predeterimined.py](https://github.com/jacobazoulay/bipedal-robot-rl/blob/main/src/models/predeterimined.py "predeterimined.py"): manually hardcoded policy for baseline
		-  [random.py](https://github.com/jacobazoulay/bipedal-robot-rl/blob/main/src/models/random.py "random.py"): random policy for baseline
- **utils**: contains plotting util

## Project Details
### Rendered Environment
![Robot rendered environment](/assets/images/robot_env.png)

### DQN Architecture
![DQN architecture](/assets/images/dqn_arch.png)

### ARS Result
![ARS reward](/assets/images/ARS_reward.png)
