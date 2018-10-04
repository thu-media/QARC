import os
import tensorflow as tf
import tflearn
import numpy as np
FEATURE_NUM = 64
KERNEL = 4

class InnovationNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        self.inputs, self.out = self.create_network()
        self.crossloss = tflearn.objectives.categorical_crossentropy(self.out, self.acts)
        self.optimize = tf.train.AdamOptimizer(self.lr_rate).minimize(self.crossloss)
        self.sess.run(tf.global_variables_initializer())

    def create_network(self):
        with tf.variable_scope('innovation'):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])
            split_array = []
            for i in xrange(self.s_dim[0] - 1):
                split = tflearn.conv_1d(inputs[:, i:i + 1, :], FEATURE_NUM, KERNEL, activation='relu')
                flattern = tflearn.flatten(split)
                split_array.append(flattern)
            
            #dense_net= tflearn.fully_connected(inputs[:, -1:, 0:5], FEATURE_NUM, activation='relu')
            split_array.append(inputs[:, -1, 0:5])
            merge_net = tflearn.merge(split_array, 'concat')
            dense_net_0 = tflearn.fully_connected(merge_net, 64, activation='relu')

            out = tflearn.fully_connected(dense_net_0, self.a_dim, activation='softmax')

            return inputs, out

    def loss(self,inputs,action):
        inputs = np.reshape(inputs,(-1,self.s_dim[0], self.s_dim[1]))
        acts = np.zeros([1, self.a_dim])
        acts[0,action] = 1
        return self.sess.run(self.crossloss, feed_dict={
            self.inputs: inputs,
            self.acts: acts
        })

    def train(self, inputs, action):
        inputs = np.reshape(inputs,(-1,self.s_dim[0], self.s_dim[1]))
        acts = np.zeros([1, self.a_dim])
        acts[0,action] = 1
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.acts: acts
        })

    def predict(self, inputs):
        inputs = np.reshape(inputs,(-1,self.s_dim[0], self.s_dim[1]))
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })