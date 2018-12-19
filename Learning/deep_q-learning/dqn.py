import tensorflow as tf
import numpy as np
from collections import deque
from tqdm import tqdm
from connect import Connect
import matplotlib.pyplot as plt


class DQN:
    def __init__(self, training=False, tensorboard=False, graph=False, lr=1e-5, gamma=0.95, batch=5000, sample=100, epis=250):
        # HYPER PARAMETERS
        self.LEARNING_RATE = lr
        self.GAMMA = gamma
        self.EXPLORE_START = 0.95
        self.EXPLORE_STOP = 0.01
        self.DECAY_RATE = 1e-4
        self.BATCH_SIZE = batch
        self.SAMPLE_SIZE = sample
        self.N_EPISODES = epis

        self.NAME = f"lr({self.LEARNING_RATE})_gamma({self.GAMMA})_batch({self.BATCH_SIZE})_sample({self.SAMPLE_SIZE})"
        print(self.NAME)

        self.memory = deque(maxlen=100000)

        self.last_dist = 0

        self.conn = Connect()
        self.create_dqn()
        print("Model successfully set up!")

        if training:
            if tensorboard: 
                self.tensorboard()
            reward, loss = self.train(tensorboard)
            if graph:
                self.graph(reward, loss)
        else:
            self.run()

        self.conn.close()

    def create_dqn(self):
        tf.reset_default_graph()

        with tf.name_scope("Placeholders"):
            self.sensors = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="Sensors")
            self.actions = tf.placeholder(dtype=tf.float32, shape=[None], name="Actions")
            self.target_q = tf.placeholder(dtype=tf.float32, shape=[None], name="Target_Q")
            self.distance = tf.placeholder(dtype=tf.float32, shape=(), name="Distance")

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
            self.opt = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)

    def fill_memory(self):
        state, reward, done = self.step(self.rand_act())

        for _ in tqdm(range(self.BATCH_SIZE)):
            action = self.rand_act()
            next_state, reward, done = self.step(action)

            if done:
                next_state = np.zeros(state.shape)
                self.store((state, action, reward, next_state))
                state, reward, done = self.step(self.rand_act())
            else:
                self.store((state, action, reward, next_state))
                state = next_state

        while not done:
            state, _, done = self.step(self.rand_act())

        return state

    def train(self, tensorboard):
        state = self.fill_memory()
        print("Successfully filled memory, staring training...")

        rewards_list = []
        loss_list = []

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if tensorboard:
                self.writer.add_graph(sess.graph)

            record = 0
            step = 0
            for episode in range(1, self.N_EPISODES):
                total_reward = 0
                crash = False

                while not crash:
                    step += 1
                    explore = self.EXPLORE_STOP + (self.EXPLORE_START - self.EXPLORE_STOP)*np.exp(-self.DECAY_RATE*step)
                    if explore > np.random.rand():
                        action = self.rand_act()
                    else:
                        action = sess.run(self.output, feed_dict={
                            self.sensors: [state]
                        })
                    next_state, reward, done = self.step(action)
                    total_reward += reward

                    if done:
                        next_state = np.zeros(state.shape)

                        print('Episode: {}/{}'.format(episode, self.N_EPISODES),
                              'Reward: {}'.format(int(total_reward)),
                              'Loss: {:.1f}'.format(loss),
                              'Explore: {}%'.format(int(explore*100))
                              )

                        loss_list.append(loss)
                        rewards_list.append(total_reward)

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

                    targets = rewards + self.GAMMA * np.max(target_q, axis=1)
                    target_list = targets.ravel()

                    loss, _ = sess.run([self.loss, self.opt], feed_dict={
                        self.sensors: states,
                        self.actions: actions,
                        self.target_q: target_list
                    })

                    if crash and tensorboard:
                        summary = sess.run(self.write_op, feed_dict={
                            self.sensors: states,
                            self.actions: actions,
                            self.target_q: target_list,
                            self.distance: total_reward
                        })
                        self.writer.add_summary(summary, episode)
                        self.writer.flush()

                if total_reward > record:
                    record = total_reward
                    #self.save(sess)

        return rewards_list, loss_list

    def run(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.load(sess)

            state = self.recv()
            while True:
                action = sess.run(self.output, feed_dict={
                    self.sensors: [state]
                })
                state, _, _ = self.step(action)

    def store(self, data):
        self.memory.append(data)

    def sample(self):
        space = np.random.choice(np.arange(len(self.memory)),
                                 size=int(self.SAMPLE_SIZE),
                                 replace=False)
        return [self.memory[i] for i in space]

    def tensorboard(self):
        self.writer = tf.summary.FileWriter(f"./logs/{self.NAME}")
        tf.summary.scalar("Q-value", self.Q)
        #tf.summary.scalar("Q-target", self.target_q)
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Distance", self.distance)
        self.write_op = tf.summary.merge_all()

    def graph(self, reward, loss):
        epis = []
        for i in range(self.N_EPISODES-1):
            epis.append(i)
        plt.figure(1)
        plt.plot(epis, reward)
        plt.title("Reward")
        plt.ylabel("Reward")
        plt.xlabel("Episode")
        plt.axis([0, self.N_EPISODES, 0, np.max(reward)])
        plt.figure(2)
        plt.plot(epis, loss)
        plt.title("Loss")
        plt.ylabel("Loss")
        plt.xlabel("Episode")
        plt.axis([0, self.N_EPISODES, 0, np.max(loss)])
        plt.show()

    def save(self, sess):
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./models/model5.ckpt")
        print("Model saved in path: %s" % save_path)

    def load(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, "./models/model3.ckpt")
        print("Model loaded")

    def rand_act(self):
        return np.random.rand()*2 - 1

    def recv(self):
        s, _, _ = self.conn.receive()
        return s

    def step(self, action):
        self.conn.send(action)
        s, d, c = self.conn.receive()
        if c:
            self.last_dist = 0
        return s, d-self.last_dist, c
