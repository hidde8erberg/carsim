import tensorflow as tf

from connect import Connect
from network import Network
import utils

# disables stupid warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    net = Network()
    print("Model successfully initialized...")

    conn = Connect()

    net.sess.run(tf.global_variables_initializer())
    while True:
        sensors, distance, crash = utils.sort_server(conn.receive())
        feed_dict = {
            net.sensors: sensors
            # net.reward: distance
        }
        x = net.sess.run(net.output, feed_dict)
        conn.send(x)
        print(x)
