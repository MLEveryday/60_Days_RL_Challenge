import gym
import random
from collections import namedtuple
import collections
import numpy as np
import matplotlib.pyplot as plt


def select_eps_greedy_action(table, obs, n_actions):
	'''
	Select the action using a ε-greedy policy (add a randomness ε for the choice of the action)
	'''
	value, action = best_action_value(table, obs)
	
	if random.random() < epsilon:
		return random.randint(0, n_actions - 1)
	else:
		return action


def select_greedy_action(table, obs, n_actions):
	'''
	Select the action using a greedy policy (take the best action according to the policy)
	'''
	value, action = best_action_value(table, obs)
	return action


def best_action_value(table, state):
	'''
	Exploring the table, take the best action that maximize Q(s,a)
	'''
	best_action = 0
	max_value = 0
	for action in range(n_actions):
		if table[(state, action)] > max_value:
			best_action = action
			max_value = table[(state, action)]
	
	return max_value, best_action


def Q_learning(table, obs0, obs1, reward, action):
	'''
	Q-learning. Update Q(obs0,action) according to Q(obs1,*) and the reward just obtained
	'''
	
	# Take the best value reachable from the state obs1
	best_value, _ = best_action_value(table, obs1)
	
	# Calculate Q-target value
	Q_target = reward + GAMMA * best_value
	
	# Calculate the Q-error between the target and the previous value
	Q_error = Q_target - table[(obs0, action)]
	
	# Update Q(obs0,action)
	table[(obs0, action)] += LEARNING_RATE * Q_error


def test_game(env, table, n_actions):
	'''
	Test the new table playing TEST_EPISODES games
	'''
	reward_games = []
	for _ in range(TEST_EPISODES):
		obs = env.reset()
		rewards = 0
		while True:
			# Act greedly
			next_obs, reward, done, _ = env.step(select_greedy_action(table, obs, n_actions))
			obs = next_obs
			rewards += reward
			
			if done:
				reward_games.append(rewards)
				break
	
	return np.mean(reward_games)


# Some hyperparameters..
GAMMA = 0.95

# NB: the decay rate allow to regulate the Exploration - Exploitation trade-off
#     start with a EPSILON of 1 and decay until reach 0
epsilon = 1.0
EPS_DECAY_RATE = 0.9993

LEARNING_RATE = 0.8

# .. and constants
TEST_EPISODES = 100
MAX_GAMES = 15001

# Create the environment
# env = gym.make('Taxi-v2')
env = gym.make("FrozenLake-v0")
obs = env.reset()

obs_length = env.observation_space.n
n_actions = env.action_space.n

reward_count = 0
games_count = 0

# Create and initialize the table with 0.0
table = collections.defaultdict(float)

test_rewards_list = []

while games_count < MAX_GAMES:
	
	# Select the action following an ε-greedy policy
	action = select_eps_greedy_action(table, obs, n_actions)
	next_obs, reward, done, _ = env.step(action)
	
	# Update the Q-table
	Q_learning(table, obs, next_obs, reward, action)
	
	reward_count += reward
	obs = next_obs
	
	if done:
		epsilon *= EPS_DECAY_RATE
		
		# Test the new table every 1k games
		if games_count % 1000 == 0:
			test_reward = test_game(env, table, n_actions)
			print('\tEp:', games_count, 'Test reward:', test_reward, np.round(epsilon, 2))
			
			test_rewards_list.append(test_reward)
		
		obs = env.reset()
		reward_count = 0
		games_count += 1

# Plot the accuracy over the number of steps
plt.figure(figsize=(20, 10))
plt.xlabel('Steps')
plt.ylabel('Accurracy')
plt.plot(test_rewards_list)
plt.show()
