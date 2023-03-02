# Local imports
from agent import Agent

# Library imports
import gym
import numpy as np

env = gym.make('CartPole-v0')
N = 20
batch_size = 5
n_epochs = 4
alpha = 0.0003
agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, linear=True)

n_games = 100

best_score = env.reward_range[0]
score_history = []

learn_iters = 0
avg_score = 0
n_steps = 0

for i in range(n_games):
    observation = env.reset()[0]
    done = False
    score = 0
    ep_steps = 0
    while not done and ep_steps < 200:
        ep_steps += 1
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, _, _ = env.step(action)
        n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score

    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'best score: %.1f' % best_score, 'time_steps', n_steps, 'learning_steps', learn_iters)