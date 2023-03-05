# Library imports
import matplotlib.pyplot as plt
import json

metrics_file = {}

with open('./models/ddqn/saved/metrics.json') as file:
    contents = file.read()
    metrics_file = json.loads(contents)

scores = metrics_file['scores']

plt.plot(metrics_file['steps'], metrics_file['scores'], label='Scores')
plt.plot(metrics_file['steps'], metrics_file['avg_scores'], label='Average Score (over previous 50 episodes)')
plt.title('Reward over time (DDQN)')
plt.xlabel('Total Steps')
plt.ylabel('Score')
plt.legend()
plt.show()

plt.plot(range(len(metrics_file['scores'])), metrics_file['scores'], label='Scores')
plt.plot(range(len(metrics_file['scores'])), metrics_file['avg_scores'], label='Average Score (over previous 50 episodes)')
plt.title('Reward over time (DDQN)')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()
plt.show()

plt.plot(range(len(metrics_file['losses'])), metrics_file['losses'], label='MSE Loss')
plt.title('Loss over time (DDQN)')
plt.xlabel('Training step')
plt.ylabel('Score')
plt.legend()
plt.show()


# Figure out what epsilon looks like
epsilon_delta = 25e-7
epsilons = [1 - epsilon_delta * steps for steps in metrics_file['steps']]
plt.plot(metrics_file['steps'], epsilons, label='Epsilon')
plt.title('Epsilon over time')
plt.xlabel('Training step')
plt.ylabel('Epsilon')
plt.legend()
plt.show()
