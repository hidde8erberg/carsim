import tensorflow as tf
import numpy as np


class Network:
    def __init__(self, lr=0.01, gamma=0.95):
        self.lr = lr
        self.gamma = gamma

        np.random.seed(1)
        tf.set_random_seed(1)

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.build_net()

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        # https://github.com/ashutoshkrjha/Cartpole-OpenAI-Tensorflow/blob/master/cartpole.py
        # self.probs = tf.nn.softmax(logits=self.output)
        # self.loss = tf.reduce_mean(self.reward * self.probs)
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # self.train_op = self.optimizer.minimize(self.loss)

    def build_net(self):
        with tf.name_scope('inputs'):
            self.sensors = tf.placeholder(dtype=tf.float32, shape=[None, 5])
            self.acts = tf.placeholder(dtype=tf.float32, shape=[None, 1])
            self.reward = tf.placeholder(dtype=tf.float32, shape=[None])

        layer1 = tf.layers.dense(self.sensors, 50, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(layer1)
        layer2 = tf.layers.dense(dropout1, 50, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(layer2)
        layer3 = tf.layers.dense(dropout2, 50, activation=tf.nn.relu)
        dropout3 = tf.layers.dropout(layer3)
        self.output = tf.layers.dense(dropout3, 1)

        #self.sig_act_prop = tf.nn.sigmoid(output)

        with tf.name_scope('loss'):
            #log_prob = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.acts)
            loss = tf.reduce_mean(self.output * self.reward)

        with tf.name_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(loss)

    def act(self, observation):
        action = self.sess.run(self.output, feed_dict={self.sensors: observation[:]})
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        self.sess.run(self.train_op, feed_dict={
            self.sensors: np.vstack(self.ep_obs),
            self.acts: np.array(self.ep_as),
            self.reward: discounted_ep_rs_norm
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0

        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "models/model.ckpt")
        print("Model saved in path: %s" % save_path)

    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, "models/model.ckpt")
        print("Model loaded")
