# Usage: python JH_model_fitting.py /src/data/wind_sensing/apparent_wind_visual_feedback/sw_dist_logstep_wind_0.001_debug_yes_vec_norm_train_actor_std/eval/plume_14421_37e2cd4be4c96943d0341849b40f81eb/noisy3x5b5.pkl 4 1
#  python JH_model_fitting.py dataset.pkl K_num_states seed tolerance
import glmhmm 
import numpy as np
import sys
sys.path.append('/src/tools/flytrack/model/')
from fitting import *
import glmhmm
import os
import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import tamagotchi.eval.log_analysis as log_analysis

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Fit a GLMHMM model to a dataset')
    parser.add_argument('--eval_pkl', type=str, nargs='+', help='Paths to the dataset eval logs')
    parser.add_argument('--obs_pkl', type=str, nargs='+', default=None, help='Paths to the observability eval logs')
    parser.add_argument('--dataset', type=str, nargs='+', default=None, help='The wind/plume dataset name to look for in config.data_dir, for log_analysis.get_selected_df')
    parser.add_argument('--K', type=int, help='Number of states')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--tolerance', type=float, default=None, help='Tolerance for convergence')
    parser.add_argument('--ID', type=str, default=False, help='HMMGLM_seed{seed}_{ID}_randSeed{rand_seed}_K{args.K}.npz')
    
    args = parser.parse_args()
    # Sanity check 
    # if obs_pkl or dataset is provided, make sure it is the same length as eval_pkl
    if args.obs_pkl is not None:
        assert len(args.eval_pkl) == len(args.obs_pkl), "Length of eval_pkl and obs_pkl must be the same"
    else:
        args.obs_pkl = [eval_pkl.replace(".pkl", "_observability_test.pkl") for eval_pkl in args.eval_pkl]
    if args.dataset is not None:
        assert len(args.eval_pkl) == len(args.dataset), "Length of eval_pkl and dataset must be the same"
    else:
        args.dataset = [os.path.basename(eval_pkl).replace('.pkl', '') for eval_pkl in args.eval_pkl]
    args.eval_folder = [os.path.dirname(eval_pkl) + '/' for eval_pkl in args.eval_pkl]
    # TODO: what if multiple files are not from the same seed? How to name then>? 
    args.model_name = os.path.basename(os.path.dirname(args.eval_folder[0])).split('_')[1]
    print(f"now fitting {args.K} state on {args.model_name}, {args.dataset[0]} seed {args.K}")
    if not args.ID:
        args.out_fname = f"HMMGLM_seed{args.model_name}_randSeed{args.seed}_K{args.K}.npz"
    else:
        args.out_fname = f"HMMGLM_seed{args.model_name}_{args.ID}_randSeed{args.seed}_K{args.K}.npz"
    args.out_path = os.path.join(args.eval_folder[0], args.out_fname)
    print(f"results will be saved to {args.out_path}")
    
    return args

    
args = parse_args()
np.random.seed(args.seed)

# eval_pkl = sys.argv[1]
# obs_pkl = eval_pkl.replace(".pkl", "_observability_test.pkl")
# eval_folder = os.path.dirname(eval_pkl) + '/'
# dataset = os.path.basename(eval_pkl).replace('.pkl', '')
# K = int(sys.argv[2]) # number of states
# seed = int(sys.argv[3]) # number of instances
# try:
    # tolerance = float(sys.argv[4]) # tolerance for convergence
# except IndexError:
    # tolerance = None


last_file_num_trials = 0
for i in range(len(args.obs_pkl)):
    obs_fname = args.obs_pkl[i]
    eval_fname = args.eval_pkl[i]
    eval_folder = args.eval_folder[i]
    dataset = args.dataset[i]
    with open(obs_fname, 'rb') as f_handle:
        observability_tupl = pickle.load(f_handle)
        print(len(observability_tupl))
    with open(eval_fname, 'rb') as f_handle:
        # based on open_loop_perturbation.py
        dataset = 'noisy3x5b5'
        # eval_folder = '/src/data/wind_sensing/apparent_wind_visual_feedback/sw_dist_logstep_wind_0.001_debug_yes_vec_norm_train_actor_std/eval/plume_14421_37e2cd4be4c96943d0341849b40f81eb/'
        # eval_folder = '/src/data/wind_sensing/apparent_wind_visual_feedback/sw_dist_logstep_wind_0.001_debug_yes_vec_norm_train_actor_std/eval/plume_17519_6aca800e09d4942c5d296ae7157fcf8b/'
        selected_df = log_analysis.get_selected_df(eval_folder, [dataset],
                                                n_episodes_home=240,
                                                n_episodes_other=240,  
                                                balanced=False,
                                                oob_only=False,
                                                verbose=True,
                                                log_fname=eval_fname)

        traj_df_stacked, stacked_neural_activity = log_analysis.get_traj_and_activity_and_stack_them(selected_df, 
                                                                                                    obtain_neural_activity = True, 
                                                                                                    obtain_traj_df = True, 
                                                                                                    get_traj_tmp = True,
                                                                                                    extended_metadata = True) # get_traj_tmp 
        print(traj_df_stacked.shape)
        print(stacked_neural_activity.shape)

    
        
    ls_EV_no_nan, ls_t_sim, ls_x_sim, ls_window_size, ls_eps_idx = zip(*observability_tupl)

    print(ls_eps_idx)
    # Preprocess the trajectory data
    # select episodes that have observability matrices
    eps_at = [True if ep_i in ls_eps_idx else False for ep_i in traj_df_stacked['ep_idx'] ]
    subset_traj_df_stacked = traj_df_stacked[eps_at]
    subset_stacked_neural_activity = stacked_neural_activity[eps_at]

    # for every episode, drop the last row
    subset_traj_df_stacked.reset_index(drop=True, inplace=True)
    last_rows = subset_traj_df_stacked.groupby('ep_idx').tail(1).index
    print('dropping', len(last_rows), 'rows')
    # drop the last row of each episode
    
    tmp_filtered_df = subset_traj_df_stacked.drop(index=last_rows)
    tmp_filtered_neural_activity = np.delete(subset_stacked_neural_activity, last_rows, axis=0)

    # calculate time since last wind change
        # based on /src/JH_boilerplate/agent_evaluatiion/traj_analysis_preprocess.ipynb
    tmp_filtered_df = tmp_filtered_df.groupby('ep_idx').apply(log_analysis.calc_time_since_last_wind_change).reset_index(drop=True)

    # create time column in tmp_filtered_df to match with EV_no_nan, starting from 0 to trial end 
    tmp_filtered_df['time'] = tmp_filtered_df.groupby('ep_idx')['t_val'].transform(lambda x: x - x.iloc[0])
    tmp_filtered_df['time'] = tmp_filtered_df['time'].round(2)

    # shift ep_idx by last_file_num_trials
    tmp_filtered_df['ep_idx'] += last_file_num_trials
    last_file_num_trials = traj_df_stacked['ep_idx'].nunique()
    if not i:
        filtered_df = tmp_filtered_df
        filtered_neural_activity = tmp_filtered_neural_activity
    else:
        filtered_df = pd.concat([filtered_df, tmp_filtered_df], ignore_index=True)
        filtered_neural_activity = np.concatenate([filtered_neural_activity, tmp_filtered_neural_activity], axis=0)
    print("tmp_filtered_df shape", tmp_filtered_df.shape)
    print("tmp_filtered_neural_activity shape", tmp_filtered_neural_activity.shape)

print("filtered_df shape", filtered_df.shape)
print("filtered_neural_activity shape", filtered_neural_activity.shape)

# TODO load multiple EV datasets - not considering for now
# # Preprocess the EV data 
# # stack the EV data
# ls_EV_no_nan = [df.assign(ep_idx=ep_idx) for df, ep_idx in zip(ls_EV_no_nan, ls_eps_idx)]
# EV_no_nan = pd.concat(ls_EV_no_nan)
# print(EV_no_nan.shape)
# # Merge with filtered_dfa
# EV_no_nan['time'] = EV_no_nan['time'].round(2)
# EV_no_nan = EV_no_nan.merge(filtered_df[['ep_idx', 'time', 'time_since_last_wind_change', 'odor_01']], on=['ep_idx', 'time'], how='inner')
# print(EV_no_nan.shape)

# Preprocess action data
# get speed and acceleration
obs_df = pd.DataFrame()
obs_df['ep_idx'] = filtered_df['ep_idx']
obs_df['step'] = filtered_df['step'] # squashed action
obs_df['step_dt'] = obs_df.groupby('ep_idx')['step'].diff()
obs_df['speed'] = filtered_df['step']*2 # in m/s unit
obs_df['acceleration'] = obs_df.groupby('ep_idx')['speed'].diff()
obs_df['turn'] = filtered_df['turn']
obs_df['angular_velocity_turn'] = ((obs_df['turn']  - 0.5)*2) * (6*np.pi) # in rad/s unit
print("range of turn is ", obs_df['turn'].min(), obs_df['turn'].max())
# obs_df['acceleration'] = obs_df['acceleration'].fillna(0) # TODO check timing 

# calculate angular velocity and angular acceleration
obs_df['heading_phi'] = np.angle(filtered_df['agent_angle_x']+ 1j*filtered_df['agent_angle_y'], deg=False)
obs_df['heading_phi_unwrap'] = np.unwrap(np.angle(filtered_df['agent_angle_x']+ 1j*filtered_df['agent_angle_y'], deg=False))
obs_df['angular_velocity'] = obs_df.groupby('ep_idx')['heading_phi_unwrap'].diff()
obs_df['angular_acceleration'] = obs_df.groupby('ep_idx')['angular_velocity'].diff()
# obs_df['angular_acceleration'] = obs_df['angular_acceleration'].fillna(0) # TODO check timing

# set 10% trials as test set
test_idx = np.random.choice(obs_df['ep_idx'].unique(), int(0.1*len(obs_df['ep_idx'].unique())), replace=False)
obs_df['train_test_label'] = 'train'
obs_df.loc[obs_df['ep_idx'].isin(test_idx), 'train_test_label'] = 'test'
# print how many train and test episodes
print(obs_df.groupby('train_test_label')['ep_idx'].nunique())
print(obs_df[obs_df['train_test_label']=='test'].groupby('train_test_label')['ep_idx'].unique())

input_df = pd.DataFrame()
# observations of the agent
input_df['ep_idx'] = filtered_df['ep_idx']
input_df['app_wind_x'] = filtered_df['wind_x_obs']
input_df['app_wind_y'] = filtered_df['wind_y_obs']
input_df['odor'] = filtered_df['odor_eps_log']
input_df['allo_head_phi_x'] = filtered_df['agent_angle_x']
input_df['allo_head_phi_y'] = filtered_df['agent_angle_y']
input_df['ego_drift_x'] = filtered_df['ego_course_direction_x']
input_df['ego_drift_y'] = filtered_df['ego_course_direction_y']
# possible latents to include
# input_df['min_EV_zeta'] = EV_no_nan['zeta']
# input_df['time_since_last_wind_change'] = EV_no_nan['time_since_last_wind_change']
input_df['acceleration'] = obs_df['acceleration']
input_df['angular_acceleration'] = obs_df['angular_acceleration']

# set 10% trials as test set
input_df['train_test_label'] = obs_df['train_test_label']
print(input_df.groupby('train_test_label')['ep_idx'].nunique())
print(obs_df[obs_df['train_test_label']=='test'].groupby('train_test_label')['ep_idx'].unique())
train_idx = input_df[input_df['train_test_label']=='train']['ep_idx'].unique()
# drop rows with NaN
input_df.dropna(inplace=True)
obs_df.dropna(inplace=True)


# session_length = 0
# if not session_length:
#     sess = None
# if session_length: # trim the data to have equal length sessions
#     num_sessions = Y.shape[0] // session_length
#     sess = np.arange(0, Y.shape[0], 300)
#     Y = Y[:session_length*num_sessions]
#     X = X[:session_length*num_sessions]

# K=4 # number of states # move to argv
input_names = ['app_wind_x', 'app_wind_y', 
               'odor', 
               'allo_head_phi_x', 'allo_head_phi_y', 
               'ego_drift_x', 'ego_drift_y'
               # 'min_EV_zeta', 'time_since_last_wind_change', 'acceleration', 'angular_acceleration'
               ]
D=len(input_names) # number of input features
obs_names = ['step', 'turn'] # TODO try these variables in physical units
dim_output=len(obs_names) # number of output features
covar_epsilon=1e-3

Y = obs_df[obs_names][obs_df['train_test_label']=='train'].values
X = input_df[input_names][input_df['train_test_label']=='train'].values


N=X.shape[0] # length of training data
m = glmhmm.GLMHMM(N, args.K, D, dim_output, covar_epsilon)
m.optimizer_tol = args.tolerance
print('fitting model with tolerance (default if none):', m.optimizer_tol)

A_init=m.transition_matrix
w_init=m.w
pi0_init = m.pi0
init_states_seq = m.mostprob_states(X, Y).astype(int)

import time
start = time.time()

lls_pred,A_pred,w_pred,pi0_pred = m.fit(Y,X,A_init,w_init, pi0=pi0_init, fit_init_states=True, 
                                        sess=None)
print('time taken:', time.time()-start)

np.savez(args.out_path, A_init=A_init, w_init=w_init, pi0_init=pi0_init,
         A_pred=A_pred, w_pred=w_pred, pi0_pred=pi0_pred, train_idx=train_idx, test_idx=test_idx,
         input_names=input_names, obs_names=obs_names, lls_pred=lls_pred)
