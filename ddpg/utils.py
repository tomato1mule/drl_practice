import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

import gym

def clip(x, low, high):
    return np.minimum(np.maximum(x,low),high)


class ReplayBuffer:
    def __init__(self,max_len=1e6):
        self.len = 0
        self.max_len = max_len
        self.data = deque()
        
    def __len__(self):
        return self.len
    
    def append(self,T):
        for i in range(3):
            assert type(T[i]) == np.ndarray

        
        if self.len == self.max_len:
            self.data.popleft()
            self.data.append(T)
        else:
            self.len += 1
            self.data.append(T)
    
    def sample(self, batch_size):
        batch = {}
        
        batch['state'] = np.zeros((batch_size,*self.data[0][0].shape))
        batch['next_state'] = np.zeros((batch_size,*self.data[0][1].shape))
        batch['action'] = np.zeros((batch_size,*self.data[0][2].shape))
        batch['reward'] = np.zeros((batch_size,))
        batch['done'] = np.zeros((batch_size,))
        
        idxs = np.random.randint(0,self.len,size = batch_size)
        for b,i in enumerate(idxs):
            batch['state'][b] = self.data[i][0].copy()
            batch['next_state'][b] = self.data[i][1].copy()
            batch['action'][b] = self.data[i][2].copy()
            batch['reward'][b] = self.data[i][3]
            batch['done'][b] = self.data[i][4]
            
        return batch
        










