# Local imports
from .memory import ReplayBuffer

# Library imports
import tensorflow.keras as keras
import numpy as np


def build_conv_dqn(lr, n_actions):
    model = keras.Sequential([
        keras.layers.Conv2D(32, 8, strides=(4, 4), padding='same', activation='relu'),
        keras.layers.Conv2D(64, 4, strides=(2, 2), padding='same', activation='relu'),
        keras.layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(n_actions, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
    return model


def build_linear_dqn(lr, n_actions):
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(n_actions, activation=None)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
    return model


class Agent:
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, epsilon_dec=1e-3, epsilon_end=0.01, mem_size=1000000,
                 fname='./models/ddqn/saved/agent.h5', linear=False, replace_target=1000):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size)
        self.q_eval = build_linear_dqn(lr, n_actions) if linear else build_conv_dqn(lr, n_actions)
        self.q_target = build_linear_dqn(lr, n_actions) if linear else build_conv_dqn(lr, n_actions)
        self.replace_target = replace_target
        self.replace_counter = 0

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state, verbose=0)
            action = np.argmax(actions)

        return action

    def learn(self):
        if len(self.memory.states) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval.predict(states, verbose=0)
        q_next = self.q_target.predict(next_states, verbose=0)
        q_eval = self.q_eval.predict(next_states, verbose=0)

        max_actions = np.argmax(q_eval, axis=1)

        q_target = np.copy(q_pred)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma * q_next[batch_index, max_actions] * dones

        loss = self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        self.replace_counter += 1
        if self.replace_counter % self.replace_target == 0:
            self.replace_counter = 0
            self.update_network_parameters()

        return loss

    def update_network_parameters(self):
        print('Copying weights from q_eval to q_target...')
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = keras.models.load_model(self.model_file)
