# Library imports
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.states = deque(maxlen=max_size)
        self.new_states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.dones = deque(maxlen=max_size)

    def store_transition(self, state, action, reward, new_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)
        self.dones.append(1 - int(done))

    def sample_buffer(self, batch_size):
        assert len(self.states) == len(self.new_states) == len(self.actions) == len(self.rewards) == len(self.dones)
        batch = np.random.choice(len(self.states), batch_size, replace=False)

        states = np.array(self.states)[batch]
        new_states = np.array(self.new_states)[batch]
        rewards = np.array(self.rewards)[batch]
        actions = np.array(self.actions)[batch]
        dones = np.array(self.dones)[batch]

        return states, actions, rewards, new_states, dones
