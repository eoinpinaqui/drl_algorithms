from game_environment.game import SinglePlayerGame
import os
import numpy as np

print(os.getcwd())

env = SinglePlayerGame()
state = env.reset()
done = False
total_reward = 0
frames = [env.render(mode='rgb_array')]
while not done:
    state, reward, _done, _ = env.step(env.action_space.sample())
    print(f'Player: {state[0], state[1], state[2], state[3], state[4]}, '
          f'Enemy: {state[5], state[6], state[7], state[8], state[9]}, '
          f'Missile: {state[10], state[11], state[12], state[13], state[14]}')
    total_reward += reward
    done = _done
    env.render(mode='human')

env.close()
print(total_reward)
