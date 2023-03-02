# Library imports
import matplotlib.pyplot as plt
import json

metrics_file = {}

with open('./models/random/saved/metrics.json') as file:
    contents = file.read()
    metrics_file = json.loads(contents)

scores = metrics_file['scores']

plt.plot(metrics_file['steps'], metrics_file['scores'], label='Scores')
plt.plot(metrics_file['steps'], metrics_file['avg_scores'], label='Average Score (over previous 50 episodes)')
plt.title('Reward over time')
plt.xlabel('Total Steps')
plt.ylabel('Score')
plt.legend()
plt.show()

plt.plot(range(len(metrics_file['scores'])), metrics_file['scores'], label='Scores')
plt.plot(range(len(metrics_file['scores'])), metrics_file['avg_scores'], label='Average Score (over previous 50 episodes)')
plt.title('Reward over time')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()
plt.show()
