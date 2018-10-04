import gym
import gym_qarc
import numpy as np
from RL_brain import DeepQNetwork
import os

VIDEO_BIT_RATE = [0.01, 0.3, 0.5, 0.8, 1.1, 1.4]
A_DIM = len(VIDEO_BIT_RATE)
S_INFO = 6
S_LEN = 10  # take how many frames in the past

env = gym.make('QARC-v0')

RL = DeepQNetwork(n_actions=A_DIM, n_features=S_INFO*S_LEN, learning_rate=0.001, e_greedy=0.9,
                  replace_target_iter=300, memory_size=3000,
                  e_greedy_increment=0.0002,)

total_steps = 0

os.system('mkdir results')
_file = open('test.csv', 'w')
observation = env.reset()
observation = np.reshape(observation, (S_INFO*S_LEN))
for i_episode in range(3000):
    ep_r = 0
    n_step = 0
    while True:
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        observation_ = np.reshape(observation_, (S_INFO*S_LEN))
        RL.store_transition(observation, action, reward, observation_)
        ep_r += reward
        
        if total_steps > 1000:
            RL.learn()

        if n_step == 9:
            print('Epi: ', i_episode,
                '| Ep_r: ', round(ep_r, 4),
                '| Epsilon: ', round(RL.epsilon, 2))
            _file.write(str(ep_r) + '\n')
            _file.flush()
            break

        observation = observation_
        total_steps += 1
        n_step += 1
