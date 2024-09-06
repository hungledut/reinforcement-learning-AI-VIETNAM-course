import numpy as np
import gymnasium as gym
import os
import tqdm
import matplotlib . pyplot as plt
from IPython . display import Image
from matplotlib import animation
from tqdm import tqdm

env_id = 'Taxi-v3'
env = gym.make(env_id , render_mode ='rgb_array')

def init_q_table(state_space, action_space):
    q_table = np.zeros((state_space, action_space))
    return q_table

def greedy_policy(q_table, state):
    action = np.argmax(q_table[state , :])
    return action

def epsilon_greedy_policy(q_table, state, epsilon):
    rand_n = float(np.random.uniform (0, 1))
    if rand_n > epsilon :
        action = greedy_policy(q_table, state)
    else :
        action = np.random.choice(q_table.shape [1])
    return action

n_training_episodes = 30000
n_eval_episodes = 100
lr = 0.7
max_steps = 99
gamma = 0.95
eval_seed = range(n_eval_episodes)
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005

def train(env, max_steps ,q_table ,n_training_episodes , min_epsilon ,ax_epsilon ,decay_rate ,lr ,gamma):
    for episode in tqdm(range (n_training_episodes)):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
        state , info = env.reset ()
        step = 0
        terminated = False
        truncated = False

        for step in range ( max_steps ):
            action = epsilon_greedy_policy(q_table , state , epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)
            q_table[state , action] = q_table[state , action] + lr*(reward + gamma*np.max(q_table [new_state]) - q_table[state, action])
            if terminated or truncated :
                break
            state = new_state
    return q_table

state_space = 500
action_space = 6

q_table = init_q_table(state_space, action_space)
trained_q_table = train(env ,max_steps ,q_table ,n_training_episodes ,min_epsilon ,max_epsilon ,decay_rate ,lr ,gamma)

print(q_table)