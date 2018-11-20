import time
from connect import Connect

class CarEnvironment:
  def __init__(self):
    self.conn = Connect()
    self.total_dist = 0

  """
  Reset the position of the car

  WARNING: Not implemented
  """
  def reset(self):
    state, _, _ = self.conn.receive()
    return state

  """
  Move the car according to the input parameter (value between 0 and 2)
  """
  def step(self, input):
    # Map value between -1 and 1
    input = input - 1

    # Calculate distance travelled
    _, prev_dist, _ = self.conn.receive()
    self.conn.send(input)
    state, dist, crash = self.conn.receive()

    self.total_dist += dist
    
    print(f"State: {state}, reward: {dist - prev_dist}, total dist: {self.total_dist}, crash: {crash}, input: {input}")

    # Reset total_dist if car crashed
    if crash:
      self.total_dist = 0
    
    #Return the state, reward (delta distance * 100) and if the car crashed
    return state, (dist - prev_dist) * 100, crash