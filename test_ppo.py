# Local imports
from models.ppo.agent import Agent
from game_environment.game import SinglePlayerGame

# Library imports
from collections import deque
import numpy as np

env = SinglePlayerGame()
N = 32
batch_size = 8
n_epochs = 4
alpha = 0.00025
agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, linear=False)

agent.load_models()

for i in range(5):
    observation = env.reset()
    frame_buffer = deque([observation, observation, observation, observation], maxlen=4)
    done = False
    score = 0
    ep_steps = 0
    while not done:
        ep_steps += 1
        current_observation = np.array(frame_buffer).astype(np.float32)
        action, prob, val = agent.choose_action(current_observation)
        observation, reward, done, info = env.step(action)
        score += reward
        frame_buffer.append(observation)
        env.render()

    print(f'Test Episode {i}, Length: {ep_steps}, Score: {score}')