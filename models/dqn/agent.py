# Local imports
from .memory import ReplayBuffer

# Library imports
import tensorflow.keras as keras
import numpy as np


class LinearNetwork(keras.Model):
    def __init__(self, n_actions):
        super(LinearNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(256, activation='relu')
        self.dense2 = keras.layers.Dense(256, activation='relu')
        self.dense3 = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        q = self.dense3(x)
        return q


class ConvNetwork(keras.Model):
    def __init__(self):
        super(ConvNetwork, self).__init__()
        self.conv1 = keras.layers.Conv2D(32, 8, strides=(4, 4), padding='same', activation='relu')
        self.conv2 = keras.layers.Conv2D(64, 4, strides=(2, 2), padding='same', activation='relu')
        self.conv3 = keras.layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(512, activation='relu')
        self.dense2 = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        q = self.dense2(x)
        return q


class Agent:
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, epsilon_dec=1e-3, epsilon_end=0.01, mem_size=1000000, fname='./models/dqn/saved/weights/', linear=False):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size)

        self.q_eval = LinearNetwork(n_actions) if linear else ConvNetwork(n_actions)
        self.q_eval.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            state = np.array([observation])
            actions = self.q_eval(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if len(self.memory.states) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval(states).numpy()
        q_next = self.q_eval(next_states).numpy()

        q_target = np.copy(q_pred)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * dones

        loss = self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        return loss

    def save_model(self):
        self.q_eval.save_weights(self.model_file)

    def load_model(self):
        self.q_eval.load_weights(self.model_file)
