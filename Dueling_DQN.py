import torch
import copy
from collections import deque
import numpy as np
import random
import torch.nn as nn
from config import args
from model import Model_dueling
import torch.nn.functional as F

class Dueling_DQNAgent():
  """
  Class that defines the functions required for training the DQN agent
  """
  def __init__(self, n_states, n_actions, batch_size= args.batch_size, gamma= args.gamma, eps_max= args.eps_max, lr= args.lr, N= args.N, eps_end= args.eps_end, eps_decay= args.eps_decay):
      
    self.gamma= gamma
    self.n_states= n_states
    self.n_actions= n_actions
    
    # for epsilon-greedy exploration strategy
    self.eps_max= eps_max
    self.eps_decay= eps_decay
    self.eps_end= eps_end
    self.lr= lr
    
    self.memory= N
    self.batch_size= batch_size

    # instances of the network for current policy and its target
    self.q_net= Model_dueling(self.n_states, self.n_actions)
    self.tg_net= copy.deepcopy(self.q_net)

    self.criteria = torch.nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

    # instance of the replay buffer
    self.buffer = deque(maxlen=self.memory)

  def insert(self, state, action, reward, next_state, done):

    """
    Parameters
    ----
    s: float
      State value
    a: int
      Action value
    r: float
      reward value
    s': float
      next state value
    
    device: str
      Name of the device (cuda or cpu) on which the computations would be performed
    
    sars= (s, a, r, s') this is a transition that will be store in the Buffer.
    done is a bool variable, that tells us whether a state is termial or not.
    
    """
    state= np.expand_dims(state, 0)
    next_state= np.expand_dims(next_state, 0)
        
    self.buffer.append((state, action, reward, next_state, done))

  def buffer_size(self):
    """
      This function return the size of the Replay Buffer.
    """

    return len(self.buffer)

  def sample_Buffer(self, m):
    """
      Function to pick 'm' samples from the memory that are selected uniformly at random, such that m = batchsize

      Parameters
      ---
      batchsize: int
          Number of elements to randomly sample from the memory in each batch
      device: str
          Name of the device (cuda or cpu) on which the computations would be performed

      Returns
      ---
      Tensors representing a batch of transitions sampled from the memory
    """
    state, action, reward, next_state, done = zip(*random.sample(self.buffer, m))
    return np.concatenate(state), action, reward, np.concatenate(next_state), done


  def train(self):

    s_batch, a_batch, r_batch, s_n_batch, done_bacth = self.sample_Buffer(self.batch_size)

    s_batch= args.Variable(torch.FloatTensor(np.float32(s_batch)))
    a_batch= args.Variable(torch.LongTensor(a_batch))

    s_n_batch= args.Variable(torch.FloatTensor(np.float32(s_n_batch)), volatile=True)
    r_batch= args.Variable(torch.FloatTensor(r_batch))

    done_bacth= args.Variable(torch.FloatTensor(done_bacth))

    ## We 
    q_values= self.q_net.policy(s_batch)

    # We Compute the expected Q values by using target Network
    next_q_values= self.tg_net.policy(s_n_batch)

    q_value= q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)
    next_q_value= next_q_values.max(1)[0]

    expected_q_value= r_batch + self.gamma * next_q_value * (1 - done_bacth) # Target from Bellmann Equation

    ## Loss Computation
    # loss= agent.criteria(q_value, expected_q_value).mean()
    loss= F.smooth_l1_loss(q_value, expected_q_value).mean()

    ## ---- Optimization step ------- ##
    self.optimizer.zero_grad()
    loss.backward(retain_graph=True)
    #nn.utils.clip_grad_norm(self.q_net.parameters(), args.grad_clip)
    self.optimizer.step()
    ## ---- Optimization step ------- ##
    return loss