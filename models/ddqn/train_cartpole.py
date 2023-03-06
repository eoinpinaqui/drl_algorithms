# Local imports
from agent import Agent

# Library imports
import gym
import numpy as np
import tensorflow as tf
import random

# Set the seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

env = gym.make('CartPole-v0')
agent = Agent(lr=0.001, gamma=0.95, n_actions=2, epsilon=1.0, batch_size=64, epsilon_end=0.0, replace_target=100, linear=True)

n_games = 100
epsilon_delta = 1.0 / (n_games / 2)

best_score = env.reward_range[0]
score_history = []
avg_score = 0
n_steps = 0

for i in range(n_games):
    observation = env.reset()[0]
    done = False
    score = 0
    ep_steps = 0
    while not done and ep_steps < 200:
        ep_steps += 1
        n_steps += 1
        action = agent.choose_action(observation)
        next_state, reward, done, _, _ = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, next_state, int(done))
        observation = next_state
        agent.learn()

    agent.decrease_epsilon(epsilon_delta)
    score_history.append(score)
    avg_score = np.mean(score_history[-10:])

    print(f'Episode {i}, Score: {score}, Average Score: {avg_score} Time steps: {n_steps}, Epsilon: {agent.epsilon}')