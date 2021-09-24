from numpy.core.einsumfunc import _optimal_path
import torch.nn as nn
import torch
import random
from config import args

class Model(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Model, self).__init__()
        self.n_actions= n_actions
        self.n_states= n_states
        
        self.MLP = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions)
        )
        
    def policy(self, x):
        return self.MLP(x)
    
    def selection_action(self, state, epsilon):
      """
        This epsilon is for eps-greedy plociy
      """
      q=0
      if random.random() > epsilon:
        with torch.no_grad():
          state= args.Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
          q_value= self.policy(state)
          action= q_value.max(1)[1].data[0].item() 
          q= q_value.max(1)[0].data[0].item()
      else:
          action= random.randrange(self.n_actions)
      return action, q
  

class Model_dueling(nn.Module):
  
  def __init__(self, n_states, n_actions):
      super(Model_dueling, self).__init__()
      self.n_actions= n_actions
      self.n_states= n_states
      
      
      self.feature = nn.Sequential(
          nn.Linear(self.n_states, 256),
          nn.ReLU()
      )
      
      self.advantage = nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Linear(256, self.n_actions)
      )
      
      self.value = nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Linear(256, 1)
      )
      
  def policy(self, x):
      x = self.feature(x)
      advantage = self.advantage(x)
      value     = self.value(x)
      return value + advantage  - advantage.mean()
  
  def selection_action(self, state, epsilon):

      # """
      #   This epsilon is for eps-greedy plociy
      # """
    q=0
    if random.random() > epsilon:

      state= args.Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
      q_value= self.policy(state)
      action= q_value.max(1)[1].data[0].item() 
      q= q_value.max(1)[0].data[0].item()
    else:

      action= random.randrange(self.n_actions)
    return action, q
  

  
  
  