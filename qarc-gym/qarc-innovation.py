import gym
import gym_qarc
import numpy as np
from inet import InnovationNetwork
import os
import tensorflow as tf

VIDEO_BIT_RATE = [0.01, 0.3, 0.5, 0.8, 1.1, 1.4]
A_DIM = len(VIDEO_BIT_RATE)
S_INFO = 6
S_LEN = 10  # take how many frames in the past
LR_RATE = 1e-4
RAND_RANGE = 1000

env = gym.make('QARC-v0')
total_steps = 0

os.system('mkdir results')
_file = open('test.csv', 'w')
observation = env.reset()
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    _innovation_network = InnovationNetwork(
        sess, state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=LR_RATE)
    for i_episode in range(3000):
        total_max_reward = 0.0
        total_reward = 0.0

        for _step in range(10):
            _reward = -1000
            _action = -1
            for action in range(A_DIM):
                observation_, reward, done, info = env.step_without_change(
                    action)
                if reward > _reward:
                    _reward, _action = reward, action
            _pred = _innovation_network.predict(observation)[0]
            action_cumsum = np.cumsum(_pred)
            _selected = (action_cumsum > np.random.randint(
                1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            observation_, reward, done, info = env.step(_selected)
            print _action, _selected
            _innovation_network.train(observation, _action)  # ground truth

            total_max_reward += _reward
            total_reward += reward

            observation = observation_

        print i_episode, total_reward, total_max_reward
        _file.write(str(i_episode) + ',' + str(total_reward) +
                    ',' + str(total_max_reward) + '\n')
        _file.flush()
    _file.close()
