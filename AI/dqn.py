import tensorflow as tf 
import numpy as np 

class DQN:
    def __init__(self):
        self.lr = 0.001
        self.gamma = 0.95

        self.init_net()

    def init_net(self):
        with tf.name_scope('inputs'):
            self.sensors = tf.placeholder(dtype=tf.float32, shape=[None, 5])
            self.actions = tf.placeholder(dtype=tf.float32, shape=[None, 1])
            self.reward = tf.placeholder(dtype=tf.float32, shape=[None])

        layer1 = tf.layers.dense(self.sensors, 50, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(layer1)
        layer2 = tf.layers.dense(dropout1, 50, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(layer2)
        layer3 = tf.layers.dense(dropout2, 50, activation=tf.nn.relu)
        dropout3 = tf.layers.dropout(layer3)
        output = tf.layers.dense(dropout3, 1)

        self.softmax = tf.nn.softmax(output)

        with tf.name_scope('loss'): 
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = output, labels = self.actions)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.reward) 

        with tf.name_scope('train'):
             self.train_opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
