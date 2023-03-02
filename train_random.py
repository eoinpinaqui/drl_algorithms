# Local imports
from game_environment.game import SinglePlayerGame

# Library imports
import numpy as np
import json

env = SinglePlayerGame()

n_max_episodes = 10000

score_history = []
step_history = []
avg_score_history = []

n_steps = 0
best_score = 0

for i in range(n_max_episodes):
    observation = env.reset()
    done = False
    score = 0
    ep_steps = 0
    while not done:
        ep_steps += 1
        n_steps += 1
        state, reward, _done, _ = env.step(env.action_space.sample())
        score += reward
        done = _done
    score_history.append(score)
    step_history.append(n_steps)
    avg_score = np.mean(score_history[-50:])
    avg_score_history.append(avg_score)

    if avg_score > best_score and i > 50:
        best_score = avg_score

    print(f'Episode {i}, Length: {ep_steps}, Score: {score}, Average Score: {avg_score}')

print(f'Best Score: {best_score}')
with open('./models/random/saved/metrics.json', 'w') as outfile:
    outfile.write(json.dumps({
        'scores': score_history,
        'avg_scores': avg_score_history,
        'steps': step_history
    }, indent=4))

