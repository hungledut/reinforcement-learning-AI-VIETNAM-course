import numpy as np
import gym
import os
import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from IPython.display import Image
from matplotlib import animation
# from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the environment
env_id = 'LunarLander-v2'
env = gym.make(env_id , new_step_api = True )

state_space = env.observation_space.shape[0]
print('State Space:', state_space)
action_space = env.action_space.n
print('Action Space:', action_space)

# Policy Network
class Policy(nn.Module):
    def __init__(self , s_size , a_size , h_size ):
        super (Policy , self ).__init__ ()
        self.fc1 = nn.Linear( s_size , h_size )
        self.fc2 = nn.Linear( h_size , h_size * 2)
        self.fc3 = nn.Linear( h_size * 2, a_size )
    def forward(self , x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim =1)
    def act(self, state ):
        state = torch.from_numpy(state).float().unsqueeze(0)  #.to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item() , m.log_prob(action)
    

# Training Function
def reinforce(
        policy ,
        optimizer ,
        n_training_episodes ,
        max_steps ,
        gamma ,
        print_every
        ):
    scores_deque = deque( maxlen =100)
    scores = []
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset ()

        for t in range(max_steps):
            action , log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state , reward , done , _, info = env.step(action)
            rewards.append(reward)
            if done :
                break
        scores_deque.append(sum( rewards ))
        scores.append(sum( rewards ))

        returns = deque( maxlen = max_steps )
        n_steps = len( rewards )

        for t in range( n_steps )[:: -1]:
            disc_return_t = returns[0] if len( returns ) > 0 else 0
            returns.appendleft( gamma*disc_return_t + rewards[t])

        eps = np.finfo(np.float32 ).eps.item()

        returns = torch.tensor( returns )
        returns = ( returns - returns.mean()) / ( returns.std() + eps)

        policy_loss = []
        for log_prob , disc_return in zip( saved_log_probs , returns ):
            policy_loss.append(-log_prob * disc_return )
        policy_loss = torch.cat( policy_loss ).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(" Episode {}\ tAverage Score : {:.2 f}".format( i_episode , np.mean(scores_deque )))
    return scores

h_size = 128
lr = 0.001

policy = Policy (
        s_size = state_space ,
        a_size = action_space ,
        h_size = h_size ,
        ).to( device )
optimizer = optim.Adam( policy.parameters() , lr=lr)

n_training_episodes = 20
max_steps = 10
gamma = 0.99

scores = reinforce (
        policy ,
        optimizer ,
        n_training_episodes ,
        max_steps ,
        gamma ,
        print_every = 100)

    

