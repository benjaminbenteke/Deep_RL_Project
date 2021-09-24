"""Add all arguments to this file"""
import argparse
import torch

USE_CUDA = torch.cuda.is_available()
args = argparse.Namespace(
    num_episodes = 100,
    steps_done = 0,
    TARGET_UPDATE = 10,              
    manualSeed = 999,
    eps_max= 0.5,
    eps_end= 0.05,
    eps_decay= 500,
    batch_size= 32,
    gamma= 0.99,
    lr = 1e-3,
    N= 100000,
    grad_clip = 1.0,
    Variable = lambda *args, **kwargs: torch.autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else torch.autograd.Variable(*args, **kwargs),
    reward_threshold= 475,
    min_episodes_criterion= 500

)