import numpy as np
import pandas as pd
import pickle
from model.util import *

# Load the data from the pickle file
with open("data/constantx5b5.pkl", "rb") as file:
    data = pickle.load(file)

colors = ['b','r','g', 'y', 'm', 'k', 'grey', 'orange']

np.squeeze(data[0]['observations'], axis=1)
a=(np.tanh(np.array(data[0]['actions'])) + 1)/2

# Get trajectories info from file
num_trajectories = len(data) # use all trajectories in the file
trajectories = [data[i]['trajectory'] for i in range(num_trajectories)] # list of trajectories(num_traj x T_traj_i x 2)
obs_normalize = [np.squeeze(data[i]['observations'], axis=1) for i in range(num_trajectories)] # list of observations(num_traj x T_traj_i x 7)
infos = [data[i]['infos'] for i in range(num_trajectories)] # list of infos (num_traj x T_traj_i x dict)
obs_raw = [ProcObs(data[i]).to_numpy() for i in range(num_trajectories)] # processed obs
actions = [(np.tanh(np.array(data[i]['actions'])) + 1)/2 \
            for i in range(num_trajectories)] # list of squashed actions for each trajectory(num_traj x T_traj_i x 2)
obs_prevactions = [adda2obs(obs_normalize[i], actions[i]) for i in range(num_trajectories)]


# shuffle the order of trajectories
original_indices = np.arange(len(trajectories))
shuffled_indices = np.random.permutation(original_indices)
shuffled_trajectories = [trajectories[i] for i in shuffled_indices]
shuffled_obs = [obs_normalize[i] for i in shuffled_indices]
shuffled_info = [infos[i] for i in shuffled_indices]
shuffled_rightobs = [obs_raw[i] for i in shuffled_indices]
shuffled_actions = [actions[i] for i in shuffled_indices]
shuffled_obs_prevactions = [obs_prevactions[i] for i in shuffled_indices]