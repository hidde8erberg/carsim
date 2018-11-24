import tensorflow as tf
from connect import Connect
from collections import deque
import numpy as np
import time


class DQN:
    def __init__(self):
        self.lr = 0.001
        self.gamma = 0.95
        self.batch_size = 100

        self.conn = Connect()
        self.memory = deque(maxlen=1000)

        self.create_dqn()
        print("Model successfully initialized!")

        print("Filling memory...")
        self.fill_memory()
        print("Done filling memory!")
        print(self.memory)

    def create_dqn(self):
        self.sensors = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        self.actions = tf.placeholder(dtype=tf.float32, shape=[None])
        self.target_q = tf.placeholder(dtype=tf.float32, shape=[None])

        layer1 = tf.layers.dense(self.sensors, 50, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(layer1)
        layer2 = tf.layers.dense(dropout1, 50, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(layer2)
        layer3 = tf.layers.dense(dropout2, 50, activation=tf.nn.relu)
        dropout3 = tf.layers.dropout(layer3)
        self.output = tf.layers.dense(dropout3, 1)

        #self.sigmoid = tf.nn.sigmoid(self.output)

        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions))
        self.loss = tf.reduce_mean(tf.square(self.target_q - self.Q))
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def fill_memory(self):
        state, reward, done = self.step(np.random.uniform(-1, 1))

        for i in range(self.batch_size):
            print(state)
            time.sleep(0.5)
            action = np.random.uniform(-1, 1)
            next_state, reward, done = self.step(action)

            if done:
                next_state = np.zeros(state.shape)
                self.store((state, action, reward, next_state))
                state, reward, done = self.step(np.random.uniform(-1, 1))
            else:
                self.store((state, action, reward, next_state))
                state = next_state

    #def train(self):


    def store(self, data):
        self.memory.append(data)

    def sample(self, batch_size):
        return

    def tensorboard(self):
        self.writer = tf.summary.FileWriter("/tensorboard/pg/1")
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Reward_mean", self.mean_reward_)
        self.write_op = tf.summary.merge_all()

    def save(self, sess):
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./models/model.ckpt")
        print("Model saved in path: %s" % save_path)

    def load(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, "./models/model.ckpt")
        print("Model loaded")

    def reward(self, d1, d2):
        return (d2 - d1) * 10

    def recv(self):
        state, _, crash = self.conn.receive()
        return state, 1, crash

    def step(self, action):
        self.conn.send(action)
        s, r, c = self.conn.receive()
        if not c:
            return s, 1, c
        else:
            return s, 0, c

    def react(self, action):
        s1, _, _ = self.conn.receive()
        self.conn.send(action)
        s2, _, _ = self.conn.receive()
        return s1, action, 1, s2
