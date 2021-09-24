from argparse import ArgumentParser
from config import args
import gym
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import plot, play, load_checkpoint, render_episode
import matplotlib.pyplot as plt

from utils import eps_greedy

parser = ArgumentParser()
parser.add_argument('-m','--model', help='This is the agent.', default="DQN", required=True)
main_args = vars(parser.parse_args())

env = gym.make('CartPole-v1')
n_states= env.observation_space.shape[0]
n_actions= env.action_space.n

if main_args["model"].lower() == 'dqn':
  from DQN import DQNAgent
  agent = DQNAgent(n_states, n_actions)
elif main_args["model"].lower() == 'dueling':
  from Dueling_DQN import Dueling_DQNAgent
  agent= Dueling_DQNAgent(n_states, n_actions)
else:
  raise NameError(f"{main_args['model']} agent is not supported")
  

q_net = agent.q_net
tg_net= agent.tg_net

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    q_net = q_net.cuda()
    tg_net= tg_net.cuda()
tg_net.load_state_dict(q_net.state_dict())


scores, avg_scores, epis_history, losses_all, q_values_all,scores_all, eps_history = [], [], [], [], [], [], []
counter= 0
scores= []
for i in range(1, args.num_episodes+1):
  score= 0
  losses= 0
  ep_len= 0
  done= False
  obs= env.reset()

  res= [eps_greedy(agent.eps_max, agent.eps_end, episode_idx, agent.eps_decay, args.num_episodes) for episode_idx in range(args.num_episodes+1)]
  epsilon= res[i]

  episode_q_values= []
  episode_scores= []
  episode_loss= []
  while not done:
  # while not done:
    ep_len += 1 
    action, q = q_net.selection_action(obs, epsilon)
    episode_q_values.append(q)
    next_state, reward, done, info = env.step(action)
    score+=reward

    ## Store experience to the memeory
    agent.insert(obs, action, reward, next_state, done)

    obs= next_state

    if agent.buffer_size() > agent.batch_size:
      loss= agent.train()
      episode_loss.append(loss.item())

  losses_all.append(np.mean(episode_loss))
  q_values_all.append(np.mean(episode_q_values))
  scores_all.append(score)
  epis_history.append(ep_len)
  eps_history.append(agent.eps_max)
  print('episode',i,'/',args.num_episodes, 'score %.2f' % score, 'Q-values %.2f' % np.mean(episode_q_values), 'epsilon %.5f' % epsilon, 'Loss_episode %.5f' % np.mean(episode_loss))

  #Update the target network, copying all weights and biases in DQN
  if i % args.TARGET_UPDATE == 0:
        tg_net.load_state_dict(q_net.state_dict())

  if score >= args.reward_threshold and ep_len >= args.min_episodes_criterion: 
      torch.save(q_net.state_dict(), 'model.ckpt')
      break
        
plot(scores_all, q_values_all, losses_all)
# Save GIF image
images = render_episode(env, q_net, 0, 1500)
image_file = 'images/cartpole-v1.gif'
# loop=0: loop forever, duration=1: play each frame for 1ms
images[0].save(
    image_file, save_all=True, append_images=images[1:], loop=0, duration=1)
