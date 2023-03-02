# Local imports
from models.ppo.agent import Agent
from game_environment.game import SinglePlayerGame

# Library imports
from collections import deque
import numpy as np
import json

env = SinglePlayerGame()
N = 32
batch_size = 8
n_epochs = 4
alpha = 0.00025
agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, linear=True)

n_total_steps = 10000000

best_score = -1000

score_history = []
step_history = []
avg_score_history = []
actor_losses = []
critic_losses = []

learn_iters = 0
avg_score = 0
n_episodes = 0
n_steps = 0

last_write = -1000

while n_steps < n_total_steps:
    observation = env.reset()
    done = False
    score = 0
    ep_steps = 0
    while not done and ep_steps < 1000:
        n_steps += 1
        ep_steps += 1
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.remember(observation, action, prob, val, reward, done)
        observation = observation_
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1

    n_episodes += 1
    score_history.append(score)
    step_history.append(n_steps)
    avg_score = np.mean(score_history[-50:])
    avg_score_history.append(avg_score)

    if n_steps - last_write >= 1000:
        last_write = n_steps
        with open('./models/ppo/saved/metrics.json', 'w') as outfile:
            outfile.write(json.dumps({
                'scores': score_history,
                'avg_scores': avg_score_history,
                'steps': step_history,
                'actor_losses': actor_losses,
                'critic_losses': critic_losses
            }, indent=4))

    if avg_score > best_score and n_episodes > 100:
        agent.save_models()
        best_score = avg_score

    print(
        f'Episode {n_episodes}, Length: {ep_steps}, Score: {score}, Average Score: {avg_score}, Best Score: {best_score}, Total time steps: {n_steps}, Learning Iterations: {learn_iters}')
