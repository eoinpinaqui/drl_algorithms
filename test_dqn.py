# Local imports
from game_environment.game import SinglePlayerGame
from models.dqn.agent import Agent

# Library imports
import numpy as np
import json

print('Here')

env = SinglePlayerGame()
agent = Agent(lr=0.001, gamma=0.95, n_actions=env.action_space.n, epsilon=0.1, epsilon_end=0.1, epsilon_dec=1e-4, batch_size=64, linear=True)
agent.load_model()

for i in range(100):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        next_state, reward, done, _ = env.step(action)
        score += reward
        observation = next_state
        env.render()
    print(f'Episode {i}, Score: {score}')

