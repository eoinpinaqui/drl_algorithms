# Local imports
from .networks import create_conv_actor_network, create_linear_actor_network, create_conv_critic_network, create_linear_critic_network
from .ppo_memory import PPOMemory

# Library imports
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np


class Agent:
    def __init__(self, n_actions, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10,
                 model_dir='./models/ppo/saved/', linear=False):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.model_dir = model_dir
        self.linear = linear

        self.actor = create_linear_actor_network(n_actions) if linear else create_conv_actor_network(n_actions)
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha))
        self.critic = create_linear_critic_network() if linear else create_conv_critic_network()
        self.critic.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha))
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        if not self.linear:
            state = state / 255.0
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        self.actor.save(self.model_dir + 'actor.h5')
        self.critic.save(self.model_dir + 'critic.h5')

    def load_models(self):
        self.actor = keras.models.load_model(self.model_dir + 'actor.h5')
        self.critic = keras.models.load_model(self.model_dir + 'critic.h5')

    def choose_action(self, observation):
        if not self.linear:
            observation = observation / 255.0
        state = tf.convert_to_tensor([observation])

        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_prob = log_prob.numpy()[0]

        return action, log_prob, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])

                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs)
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic(states)

                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    critic_loss = keras.losses.MSE(critic_value, returns)

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))

        self.memory.clear_memory()
