import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

sys.path.append('/data/users/weixuan/work')
sys.path.append('/src/tools/flytrack/model/')
sys.path.append('/src/tools/flytrack/')
from config import colors

import numpy as np
from collections import Counter
import numpy as np
from scipy.special import logsumexp

def compute_transition_probabilities(P_base, X, W_in):
    """
    Computes P(Z_t = k | Z_t-1 = j, u_t) across all t using vectorization.
    """
    if X.shape[1] + 1 == W_in.shape[1]:
        X = np.hstack([X, np.ones((X.shape[0], 1))])
    K, _ = P_base.shape
    N, D = X.shape

    P_t = np.zeros((K, K, N))
    for t in range(N):
        u_t = X[t]
        if u_t.ndim == 1:
            u_t = u_t.reshape((-1, 1))
        log_P = np.log(P_base) + (W_in @ u_t).T
        P = np.exp(log_P - logsumexp(log_P, axis=1, keepdims=True))
        P_t[:,:,t] = P
    return P_t, P_base

# find the last non-NaN element(usually used on fitting lls)
def last_non_nan(array):
    non_nan_indices = np.where(~np.isnan(array))[0]
    if len(non_nan_indices) == 0:
        return None  
    return array[non_nan_indices[-1]]


def VisTrajStates(test_trajectories, most_likely_states_list):
    assert(len(test_trajectories) == len(most_likely_states_list))
    num_trajectories = len(test_trajectories)
    fig, axes = plt.subplots(len(test_trajectories), 1, figsize=(10, 4 * len(test_trajectories)), sharex=True)
    # Plot each trajectory
    for i in range(len(test_trajectories)): # i: traj index
        ax = axes[i]
        trajectory = np.array(test_trajectories[i])
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'k*', markersize = 10)
        # print(len(trajectory))
        # print(len(most_likely_states_list[i]))
        for j in range(len(most_likely_states_list[i])):
            ax.plot(trajectory[j,0], trajectory[j,1], marker='o', linewidth=2, markersize=4, color=colors[most_likely_states_list[i][j]])
            ax.set_title(f"Test Trajectory {i + 1}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True)
            # ax.set_ylim([-1.5,1.5])

def ProcObs(episode_log):  # input one trajectory(list) eg.data[0], return a dataframe
    # traj coords
    trajectory = episode_log['trajectory']
    traj_df = pd.DataFrame(trajectory)
    traj_df.columns = ['loc_x', 'loc_y']

    # Observations & Actions
    obs = [x[0] for x in episode_log['observations']]

    obs = pd.DataFrame(obs,columns=['wind_x', 'wind_y', 'odor', 'agent_angle_x',\
                                    'agent_angle_y', 'ego_course_direction_x', 'ego_course_direction_y'])
    traj_df['wind_x_obs'] = obs['wind_x']
    traj_df['wind_y_obs'] = obs['wind_y']

    traj_df['odor_raw'] = [ record[0]['odor_obs'] for record in episode_log['infos']]

    # calc agent angle observation from info
    traj_df['agent_angle_x'] = [ record[0]['angle'][0] for record in episode_log['infos']]
    traj_df['agent_angle_y'] = [ record[0]['angle'][1] for record in episode_log['infos']]

    # calculate course direction from info
    allo_ground_velocity  = [record[0]['ground_velocity'] for record in episode_log['infos']]
    # same calc as vec2rad_norm_by_pi, except do not normalize by pi
    allocentric_course_direction_radian = [np.angle(gv[0] + 1j*gv[1], deg=False) for gv in allo_ground_velocity]
    allocentric_head_direction_radian = [np.angle(record[0]['angle'][0] + 1j*record[0]['angle'][1], deg=False) for record in episode_log['infos']]
    egocentric_course_direction_radian = np.array(allocentric_course_direction_radian) - np.array(allocentric_head_direction_radian) # leftward positive - standard CWW convention
    ego_course_direction_x, ego_course_direction_y = np.cos(egocentric_course_direction_radian), np.sin(egocentric_course_direction_radian)
    traj_df['ego_course_direction_x'] = ego_course_direction_x
    traj_df['ego_course_direction_y'] = ego_course_direction_y

    traj_df = traj_df.drop(columns=['loc_x', 'loc_y'])

    return traj_df

def adda2obs(obs, action):
  zeros_row = np.zeros((1, 2))
  modified_array1 = np.vstack((zeros_row, action[:-1]))
  result_array = np.hstack((obs, modified_array1))
  return result_array


def find_consensus_voting(clusterings):
    """Find a consensus clustering by majority voting after relabeling."""
    n_points = len(clusterings[0])
    labels_per_point = np.array(clusterings)
    
    consensus = []
    for point in range(n_points):
        label_votes = labels_per_point[:, point]
        consensus_label = Counter(label_votes).most_common(1)[0][0]
        consensus.append(consensus_label)
    
    return np.array(consensus)


