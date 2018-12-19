import numpy as np
import random
from car_environment import CarEnvironment

class QN():
  def __init__(self):
    self.action_size = 3
    self.state_size = 2000000000
    self.qtable = np.zeros((self.state_size, self.action_size))

    self.total_episodes = 10000        # Total episodes
    self.learning_rate = 0.8           # Learning rate
    self.max_steps = 10000                # Max steps per episode
    self.gamma = 0.95                  # Discounting rate

    # Exploration parameters
    self.epsilon = 1.0                 # Exploration rate
    self.max_epsilon = 1.0             # Exploration probability at start
    self.min_epsilon = 0.01            # Minimum exploration probability 
    self.decay_rate = 0.005             # Exponential decay rate for exploration prob

    self.env = CarEnvironment()

    self.train()

  
  def train(self):
    rewards = []
    all_dist = []
    total_dist = 0

    # 2 For life or until learning is stopped
    for episode in range(self.total_episodes):
        # Reset the environment
        state = self.env.reset()
        state = 0
        step = 0
        done = False
        total_rewards = 0
        
        for step in range(self.max_steps):
            # 3. Choose an action a in the current world state (s)
            ## First we randomize a number
            exp_exp_tradeoff = random.uniform(0, 1)
            
            ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            if exp_exp_tradeoff > self.epsilon:
                action = np.argmax(self.qtable[state,:])

            # Else doing a random choice --> exploration
            else:
                action = random.randint(0, 2)

            # Take the action (a) and observe the outcome state(s') and reward (r)
            s, reward, dist, done = self.env.step(action)

            x = ""
            for i in np.array(s).flatten():
              x += str(int(round(i)))

            print("State: " + x)
            new_state = int(x)            

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            self.qtable[state, action] = self.qtable[state, action] + self.learning_rate * (reward + self.gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, action])
            
            total_rewards += reward
            total_dist += dist
            
            # Our new state is state
            state = new_state
            
            # If done (if we're dead) : finish episode
            if done == True: 
                print(f"Ep ended: {episode}")
                break
            
        # Reduce epsilon (because we need less and less exploration)
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate*episode) 
        rewards.append(total_rewards)
        all_dist.append(total_dist)

        a = np.asarray(all_dist)
        np.savetxt("ql.csv", a, delimiter=',')


