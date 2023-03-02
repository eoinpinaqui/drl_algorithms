# Library imports
import tensorflow.keras as keras

'''
Actor Networks
'''


def create_conv_actor_network(n_actions):
    return keras.Sequential([
        keras.layers.Conv2D(32, 8, strides=(4, 4), padding='same', activation='relu'),
        keras.layers.Conv2D(64, 4, strides=(2, 2), padding='same', activation='relu'),
        keras.layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(n_actions, activation='softmax')
    ])


def create_linear_actor_network(n_actions):
    return keras.Sequential([
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(n_actions, activation='softmax')
    ])


'''
Critic Networks
'''


def create_conv_critic_network():
    return keras.Sequential([
        keras.layers.Conv2D(32, 8, strides=(4, 4), padding='same', activation='relu'),
        keras.layers.Conv2D(64, 4, strides=(2, 2), padding='same', activation='relu'),
        keras.layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1, activation=None)
    ])


def create_linear_critic_network():
    return keras.Sequential([
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(1, activation=None)
    ])
