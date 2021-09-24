from DQN import DQNAgent
import math
from config import args
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import torch


def eps_greedy(eps_max, eps_end, episode_idx, eps_decay, num_eps):
  return eps_end + (eps_max - eps_end) * math.exp(-1. * episode_idx / eps_decay)

def plot(rewards, q_values, losses):
    
    #clear_output(True)
    # plt.figure(figsize=(20,5))

    # plt.subplot(131)
    # plt.title('rewards for last 100 episodes')
    # plt.xlabel("Episodes")
    # plt.ylabel('rewards for last 100 episodes')
    # plt.plot(np.ones(100)*500, color= 'red')
    # plt.plot(rewards[-100:])
    # plt.grid()

    # plt.subplot(132)
    # plt.title('Q-values per episodes')
    # plt.xlabel('Episodes')
    # plt.ylabel('Q-values  per Episode')
    # plt.plot(q_values[-100:])
    # plt.grid()

    # plt.subplot(133)
    # plt.title('Loss values per episode')
    # plt.xlabel("Episodes")
    # plt.ylabel('Loss values per episode')
    # plt.plot(losses[-100:])
    # plt.grid()
    # # plt.yscale('log')
    # plt.show()
    
    plt.figure(figsize=(20,5))

    plt.subplot(131)
    plt.title('rewards for last 100 episodes')
    plt.xlabel("Rewards")
    plt.ylabel('Rewards for last 100 episodes')
    # plt.plot(np.ones(args.num_episodes)*500, color= 'red')
    plt.plot(rewards)
    plt.grid()

    plt.subplot(132)
    plt.title('Q-values per episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Q-values per Episode')
    plt.plot(q_values)
    plt.grid()

    plt.subplot(133)
    plt.title('Loss values per episode')
    plt.xlabel("Episodes")
    plt.ylabel('Loss values per episode')
    plt.plot(losses)
    plt.grid()
    # plt.yscale('log')
    plt.show()
    

def play(env, agent, model):
  for i in tqdm(range(10)):
      obs, done, rew = env.reset(), False, 0
      while (done != True) :
        action, q = model.selection_action(obs, 0)
        # A =  model.select_action(obs, env.action_space.n, epsilon = 0)
        obs, reward, done, info = env.step(action)
        rew += reward
        sleep(0.01)
        env.render()  
      print("episode : {}, reward : {}".format(i,rew)) 
      
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    
    return model
  
# from IPython import display as ipythondisplay
from PIL import Image
from pyvirtualdisplay import Display


# display = Display(visible=0, size=(400, 300))
# display.start()


def render_episode(env, model, max_steps, epsilon): 
  screen = env.render(mode='rgb_array')
  im = Image.fromarray(screen)

  images = [im]

  state = env.reset()
  reward_episode= 0
  done = False
  k = 0 # End of the episode
  for i in range(1, max_steps + 1):
      action, _ = model.selection_action(state, epsilon= 0)
      next_state, reward, done, info = env.step(action)
      reward_episode += reward
      state = next_state
      k += 1
    
      if i % 10 == 0:
        screen = env.render(mode='rgb_array')
        images.append(Image.fromarray(screen))
      
      if k > 500:
        break
      
  return images