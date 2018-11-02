import tensorflow as tf


class Network:
    def __init__(self, epochs=500, lr=0.001):
        self.epochs = epochs
        self.learning_rate = lr

        self.sess = tf.Session()

        self.sensors = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        self.acts = tf.placeholder(dtype=tf.float32, shape=[None])
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None])

        self.layer1 = tf.layers.dense(self.sensors, 50, activation=tf.nn.relu)
        self.dropout1 = tf.layers.dropout(self.layer1)
        self.layer2 = tf.layers.dense(self.dropout1, 50, activation=tf.nn.relu)
        self.dropout2 = tf.layers.dropout(self.layer2)
        self.layer3 = tf.layers.dense(self.dropout2, 50, activation=tf.nn.relu)
        self.dropout3 = tf.layers.dropout(self.layer3)
        self.output = tf.layers.dense(self.dropout3, 1)

        # https://github.com/ashutoshkrjha/Cartpole-OpenAI-Tensorflow/blob/master/cartpole.py
        self.probs = tf.nn.softmax(logits=self.output)
        self.loss = tf.reduce_mean(self.reward * self.probs)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    #https://gist.github.com/shanest/535acf4c62ee2a71da498281c2dfc4f4
    def train_step(self, obs, acts, reward):
        batch_feed = { self.sensors: obs,
                self.acts: acts,
                self.reward: reward}
        self.train_op.run(self._train, feed_dict=batch_feed)

    def policy_rollout(env, agent):
        """Run one episode."""

        observation, reward, done = env.reset(), 0, False
        obs, acts, rews = [], [], []

        while not done:

            env.render()
            obs.append(observation)

            action = agent.act(observation)
            observation, reward, done, _ = env.step(action)

            acts.append(action)
            rews.append(reward)

        return obs, acts, rews

    def save(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "models/model.ckpt")
        print("Model saved in path: %s" % save_path)

    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, "models/model.ckpt")
        print("Model loaded")
