# Local imports
from game_environment.game import SinglePlayerGame
from models.ddqn.agent import Agent

# Library imports
import numpy as np
import json

env = SinglePlayerGame()
agent = Agent(lr=0.001, gamma=0.95, n_actions=env.action_space.n, epsilon=1.0, epsilon_end=0.1, epsilon_dec=25e-7, batch_size=64, linear=True)

n_training_episodes = 20000

best_score = -1000

score_history = []
step_history = []
avg_score_history = []
losses = []

avg_score = 0
n_episodes = 0
n_steps = 0

last_write = -1000

while n_episodes < n_training_episodes:
    observation = env.reset()
    done = False
    score = 0
    ep_steps = 0
    while not done and ep_steps < 1000:
        n_steps += 1
        ep_steps += 1
        action = agent.choose_action(observation)
        next_state, reward, done, _ = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, next_state, int(done))
        observation = next_state
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)

    n_episodes += 1
    score_history.append(score)
    step_history.append(n_steps)
    avg_score = np.mean(score_history[-50:])
    avg_score_history.append(avg_score)

    if n_steps - last_write >= 1000:
        last_write = n_steps
        with open('./models/ddqn/saved/metrics.json', 'w') as outfile:
            outfile.write(json.dumps({
                'scores': score_history,
                'avg_scores': avg_score_history,
                'steps': step_history,
                'losses': losses
            }, indent=4))

    if avg_score > best_score and n_episodes > 50:
        agent.save_model()
        best_score = avg_score

    print(
        f'Episode {n_episodes}, Length: {ep_steps}, Score: {score}, Average Score: {avg_score}, Best Score: {best_score}, Total time steps: {n_steps}, Epsilon: {agent.epsilon}')
