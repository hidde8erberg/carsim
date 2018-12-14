from dqn import DQN

# disables stupid warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

lrs = [1e-3, 1e-5, 1e-6]
gammas = [0.8, 0.9, 0.99]
batchs = [500, 1000, 5000]
samples = [50, 100, 250]
episodes = 50

for lr in lrs:
    for gamma in gammas:
        for batch in batchs:
            for sample in samples:
                dqn = DQN(training=True, tensorboard=True, graph=False, lr=lr, gamma=gamma, batch=batch, sample=sample, epis=episodes)
