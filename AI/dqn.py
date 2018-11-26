import tensorflow as tf
from connect import Connect
from collections import deque
import numpy as np


class DQN:
    def __init__(self, training=False):
        self.lr = 0.00025
        self.gamma = 0.95
        self.explore_start = 1.0
        self.explore_stop = 0.01
        self.decay_rate = 0.0001
        self.batch_size = 5000
        self.episodes = 20

        self.conn = Connect()
        self.memory = deque(maxlen=1000000)

        self.create_dqn()
        print("Model successfully initialized!")

        if training:
            self.tensorboard()
            self.train()
        else:
            self.run()

    def create_dqn(self):
        with tf.name_scope("Placeholders"):
            self.sensors = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="Sensors")
            self.actions = tf.placeholder(dtype=tf.float32, shape=[None], name="Actions")
            self.target_q = tf.placeholder(dtype=tf.float32, shape=[None], name="Target_Q")

        with tf.name_scope("Network"):
            layer1 = tf.layers.dense(self.sensors, 25, activation=tf.nn.relu, name="layer1")
            dropout1 = tf.layers.dropout(layer1)
            layer2 = tf.layers.dense(dropout1, 25, activation=tf.nn.relu, name="layer2")
            dropout2 = tf.layers.dropout(layer2)
            layer3 = tf.layers.dense(dropout2, 25, activation=tf.nn.relu, name="layer3")
            dropout3 = tf.layers.dropout(layer3)
            self.output = tf.layers.dense(dropout3, 1, activation=tf.nn.tanh, name="output_layer")

        with tf.name_scope("Loss"):
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions))
            self.loss = tf.reduce_mean(tf.square(self.target_q - self.Q))

        with tf.name_scope("Train"):
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.name_scope("Accuracy"):
            self.accuracy = 1

    def fill_memory(self):
        state, reward, done = self.step(self.rand_act())

        for i in range(self.batch_size):
            action = self.rand_act()
            next_state, reward, done = self.step(action)

            if done:
                next_state = np.zeros(state.shape)
                self.store((state, action, reward, next_state))
                state, reward, done = self.step(self.rand_act())
            else:
                self.store((state, action, reward, next_state))
                state = next_state

        return state

    def train(self):
        state = self.fill_memory()
        print("Successfully filled memory, staring training...")

        rewards_list = []

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            self.writer.add_graph(sess.graph)

            step = 0
            for episode in range(1, self.episodes):
                total_reward = 0
                crash = False

                while not crash:
                    step += 1
                    explore = self.explore_stop + (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*step)
                    if explore > np.random.rand()*10:
                        action = self.rand_act()
                    else:
                        action = sess.run(self.output, feed_dict={
                            self.sensors: [state]
                        })
                    next_state, reward, done = self.step(action)
                    total_reward += reward

                    if done:
                        next_state = np.zeros(state.shape)

                        print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              #'Training loss: {:.4f}'.format(self.loss),
                              #'Explore P: {:.4f}'.format(explore)
                              )

                        rewards_list.append((episode, total_reward))
                        self.store((state, action, reward, next_state))

                        state, reward, done = self.step(self.rand_act())

                        crash = True

                    else:
                        self.store((state, action, reward, next_state))
                        state = next_state

                    batch = self.sample()
                    states = np.array([each[0] for each in batch])
                    actions = np.array([each[1] for each in batch])
                    rewards = np.array([each[2] for each in batch])
                    next_states = np.array([each[3] for each in batch])

                    target_q = sess.run(self.output, feed_dict={
                        self.sensors: next_states
                    })

                    episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                    target_q[episode_ends] = 0

                    targets = rewards + self.gamma * np.max(target_q, axis=1)
                    target_list = targets.ravel()

                    loss, _ = sess.run([self.loss, self.opt], feed_dict={
                        self.sensors: states,
                        self.actions: actions,
                        self.target_q: target_list
                    })

                summary = sess.run(self.write_op, feed_dict={
                    self.sensors: states,
                    self.actions: actions,
                    self.target_q: target_list
                })

                if episode % 1 == 0:
                    self.writer.add_summary(summary, episode)
                    self.writer.flush()

                self.save(sess)

    def run(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.load(sess)

            state, _, _ = self.recv()
            while True:
                action = sess.run(self.output, feed_dict={
                    self.sensors: [state]
                })
                state, _, _ = self.step(action)

    def store(self, data):
        self.memory.append(data)

    def sample(self):
        space = np.random.choice(np.arange(len(self.memory)),
                                 size=int(self.batch_size/10),
                                 replace=False)
        return [self.memory[i] for i in space]

    def tensorboard(self):
        self.writer = tf.summary.FileWriter("./tensorboard/pg/1")
        tf.summary.scalar("Loss", self.loss)
        self.write_op = tf.summary.merge_all()

    def save(self, sess):
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./models/model3.ckpt")
        print("Model saved in path: %s" % save_path)

    def load(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, "./models/model.ckpt")
        print("Model loaded")

    def rand_act(self):
        return np.random.rand()*2 - 1

    def reward(self, d1, d2):
        return (d2 - d1) * 10

    def recv(self):
        s, _, c = self.conn.receive()
        if not c:
            return s, 1, c
        elif c:
            return s, 0, c

    def step(self, action):
        self.conn.send(action)
        s, _, c = self.conn.receive()
        if not c:
            return s, 1, c
        elif c:
            return s, 0, c

    def react(self, action):
        s1, _, _ = self.conn.receive()
        self.conn.send(action)
        s2, _, _ = self.conn.receive()
        return s1, action, 1, s2
