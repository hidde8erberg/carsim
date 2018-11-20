#
# Code based on https://medium.freecodecamp.org/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f
#

import tensorflow as tf
import numpy as np 
from car_environment import CarEnvironment

class PGNetwork():
  def __init__(self):
    # Set to False to let the agent play
    self.training = True

    self.state_size = 5
    self.action_size = 3

    self.max_episodes = 10000
    self.learning_rate = 0.01
    self.gamma = 0.95

    self.init_networks()
    self.init_tensorboard()

    self.env = CarEnvironment()

    self.saver = tf.train.Saver()

    if self.training:
      self.train()
    else:
      self.play()

  """
  Initialize all the networks
  """
  def init_networks(self):
    with tf.name_scope("inputs"):
      self.input_ = tf.placeholder(tf.float32, [None, self.state_size], name="input_")
      self.actions = tf.placeholder(tf.int32, [None, self.action_size], name="actions")
      self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards")

      self.mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")

      with tf.name_scope("fc1"):
        self.fc1 = tf.contrib.layers.fully_connected(inputs = self.input_,
                                                num_outputs = 10,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

      with tf.name_scope("fc2"):
          self.fc2 = tf.contrib.layers.fully_connected(inputs = self.fc1,
                                                  num_outputs = self.action_size,
                                                  activation_fn= tf.nn.relu,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer())
      
      with tf.name_scope("fc3"):
          self.fc3 = tf.contrib.layers.fully_connected(inputs = self.fc2,
                                                  num_outputs = self.action_size,
                                                  activation_fn= None,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer())

      with tf.name_scope("softmax"):
          self.action_distribution = tf.nn.softmax(self.fc3)

      with tf.name_scope("loss"):
          self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.fc3, labels = self.actions)
          self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_) 
      
      with tf.name_scope("train"):
          self.train_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

  """
  Set up tensorboard
  """
  def init_tensorboard(self):
    # Setup TensorBoard Writer
    self.writer = tf.summary.FileWriter("/tensorboard/pg/1")

    ## Losses
    tf.summary.scalar("Loss", self.loss)

    ## Reward mean
    tf.summary.scalar("Reward_mean", self.mean_reward_)

    self.write_op = tf.summary.merge_all()


  def discount_and_normalize_rewards(self, episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * self.gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    
    return discounted_episode_rewards

  def train(self):
    allRewards = []
    total_rewards = 0
    maximumRewardRecorded = 0
    episode = 0
    episode_states, episode_actions, episode_rewards = [],[],[]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for episode in range(self.max_episodes):
            
            episode_rewards_sum = 0

            # Launch the game
            state = self.env.reset()
                          
            while True:
                
                # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
                action_probability_distribution = sess.run(self.action_distribution, feed_dict={self.input_: state.reshape([1,5])})
                
                action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob

                # Perform a
                new_state, reward, done = self.env.step(action)

                # Store s, a, r
                episode_states.append(state)
                            
                # For actions because we output only one (the index) we need 2 (1 is for the action taken)
                # We need [0., 1.] (if we take right) not just the index
                action_ = np.zeros(self.action_size)
                action_[action] = 1
                
                episode_actions.append(action_)
                
                episode_rewards.append(reward)
                if done:
                    # Calculate sum reward
                    episode_rewards_sum = np.sum(episode_rewards)
                    
                    allRewards.append(episode_rewards_sum)
                    
                    total_rewards = np.sum(allRewards)
                    
                    # Mean reward
                    mean_reward = np.divide(total_rewards, episode+1)
                    
                    
                    maximumRewardRecorded = np.amax(allRewards)
                    
                    print("==========================================")
                    print("Episode: ", episode)
                    print("Reward: ", episode_rewards_sum)
                    print("Mean Reward", mean_reward)
                    print("Max reward so far: ", maximumRewardRecorded)
                    
                    # Calculate discounted reward
                    discounted_episode_rewards = self.discount_and_normalize_rewards(episode_rewards)
                                    
                    # Feedforward, gradient and backpropagation
                    loss_, _ = sess.run([self.loss, self.train_opt], feed_dict={self.input_: np.vstack(np.array(episode_states)),
                                                                    self.actions: np.vstack(np.array(episode_actions)),
                                                                    self.discounted_episode_rewards_: discounted_episode_rewards 
                                                                    })
                    
    
                                                                    
                    # Write TF Summaries
                    summary = sess.run(self.write_op, feed_dict={self.input_: np.vstack(np.array(episode_states)),
                                                                    self.actions: np.vstack(np.array(episode_actions)),
                                                                    self.discounted_episode_rewards_: discounted_episode_rewards,
                                                                        self.mean_reward_: mean_reward
                                                                    })
                    
                  
                    self.writer.add_summary(summary, episode)
                    self.writer.flush()
                    
                
                    
                    # Reset the transition stores
                    episode_states, episode_actions, episode_rewards = [],[],[]
                    
                    break
                
                state = new_state
            
            # Save Model
            if episode % 25 == 0:
                self.saver.save(sess, "./models/model.ckpt")
                print("Model saved")


  def play(self):
    with tf.Session() as sess:
      self.env.reset()
      rewards = []
      
      # Load the model
      self.saver.restore(sess, "./models/model.ckpt")

      for episode in range(10):
          state = self.env.reset()
          step = 0
          done = False
          total_rewards = 0
          print("****************************************************")
          print("EPISODE ", episode)

          while True:
              

              # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
              action_probability_distribution = sess.run(self.action_distribution, feed_dict={self.input_: state.reshape([1,5])})
              #print(action_probability_distribution)
              action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob


              new_state, reward, done = self.env.step(action)

              total_rewards += reward

              if done:
                  rewards.append(total_rewards)
                  print ("Score", total_rewards)
                  break
              state = new_state
      print ("Score over time: " +  str(sum(rewards)/10))
      
