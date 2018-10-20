from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import numpy as np


class Network:
    def __init__(self, epochs=100, lr=0.001):
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epochs = epochs
        self.learning_rate = lr
        self.memory = deque(maxlen=2000)
        self.model = self.init_model()

    def init_model(self):
        model = Sequential()
        model.add(Dense(50, input_dim=5, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def run(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(1)
        run_values = self.model.predict(state)
        return np.argmax((run_values[0]))

    def replay(self):
        minibatch = random.sample(self.memory, self.epochs)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
