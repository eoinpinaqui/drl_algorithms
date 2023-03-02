# Library imports
import matplotlib.pyplot as plt
import json

metrics_file = {}

with open('./models/ppo/saved/metrics.json') as file:
    contents = file.read()
    metrics_file = json.loads(contents)

scores = metrics_file['scores']

plt.plot(metrics_file['steps'], metrics_file['scores'], label='Scores')
plt.plot(metrics_file['steps'], metrics_file['avg_scores'], label='Average Score (over previous 100 episodes)')
plt.title('Reward over time')
plt.xlabel('Total Steps')
plt.ylabel('Score')
plt.legend()
plt.show()
