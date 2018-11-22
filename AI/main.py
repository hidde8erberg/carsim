from connect import Connect
from network import Network

# disables stupid warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

net = Network()
print("Model successfully initialized...")

conn = Connect()

for epi in range(3000):

    sensors, distance, crash = conn.receive()

    while True:
        action = net.act(sensors)
        conn.send(action)

        sensors, distance, crash = conn.receive()

        net.store_transition(sensors, action, distance)

        if crash:
            ep_rs_sum = sum(net.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            print("episode:", epi, "  reward:", int(running_reward))

            vt = net.learn()
            break
