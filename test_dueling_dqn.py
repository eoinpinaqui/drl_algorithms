# Local imports
from game_environment.game import SinglePlayerGame
from models.dueling_dqn.agent import Agent

env = SinglePlayerGame()
agent = Agent(lr=0.001, gamma=0.95, n_actions=env.action_space.n, epsilon=0.1, epsilon_end=0.1, batch_size=64, linear=True)
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