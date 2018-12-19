from dqn import DQN

# disables stupid warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

lr = 1e-6
gamma = 0.9
batch = 5000
sample = 100
episode = 50

dqn = DQN(training=True, tensorboard=True, graph=False, lr=lr, gamma=gamma, batch=batch, sample=sample, epis=episodes)
