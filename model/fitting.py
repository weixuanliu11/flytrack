from sklearn.metrics import adjusted_rand_score
import numpy as np
import ssm
import matplotlib.pyplot as plt
from glmhmm import GLMHMM
import sys
sys.path.append('/data/users/weixuan/work/model/glmhmm')
from IDglmhmm import InputDrivenGLMHMM
from util import last_non_nan


import json
import os
import numpy as np



def metric_comp(storage_all, colormap = "cividis", model_type=None):
    if model_type == "inputdriven":
        metrics = ["P base element", "P base vector", "w element", "w vector", 'Win element', 'Win vector', "accuracy", "precision", "recall", "f1"]
    elif model_type is None:
        metrics = ["A element", "A vector", "w element", "w vector", "accuracy", "precision", "recall", "f1", "match rate"]
        
    fig, axes = plt.subplots(len(metrics), 3, figsize=(12, 36))
    fig.tight_layout(pad=4.0)

    # Iterate through metrics and subplots
    for idx, metric in enumerate(metrics):
        # Extract data for the current metric
        true2init = storage_all[:, :, 0, idx]
        true2pred = storage_all[:, :, 1, idx]

        # true vs true fit(state seq only) at place index 4, 5, 6, 7
        accuracy_true2tf = storage_all[:, :, 2, 6]
        precision_true2tf = storage_all[:, :, 2, 7]
        recall_true2tf = storage_all[:, :, 2, 8]
        f1_true2tf = storage_all[:, :, 2, 9]

        # Compute vmin and vmax for the current row (shared between true2init and true2pred)
        vmin = min(true2init.min(), true2pred.min())
        vmax = max(true2init.max(), true2pred.max())

        if idx == 1:
            true2trueA = storage_all[:, :, 2, idx]
            vmin = min(vmin, true2trueA.min())
            vmax = max(vmax, true2trueA.max())
        if idx == 2:
            true2truew_element = storage_all[:, :, 2, idx]
            vmin = min(vmin, true2truew_element.min())
            vmax = max(vmax, true2truew_element.max())

        if idx == 3:
            true2truew_vector = storage_all[:, :, 2, idx]
            vmin = min(vmin, true2truew_vector.min())
            vmax = max(vmax, true2truew_vector.max())

        if idx == 4:
            true2trueWin_element = storage_all[:, :, 2, idx]
            vmin = min(vmin, true2trueWin_element.min())
            vmax = max(vmax, true2trueWin_element.max())

        if idx == 5:
            true2trueWin_vector = storage_all[:, :, 2, idx]
            vmin = min(vmin, true2trueWin_vector.min())
            vmax = max(vmax, true2trueWin_vector.max())


        # Plot true-to-init
        ax_init = axes[idx, 0]
        c1 = ax_init.imshow(true2init, aspect='auto', cmap=colormap, vmin=vmin, vmax=vmax)
        ax_init.set_title(f"{metric} - True vs Init")
        fig.colorbar(c1, ax=ax_init)
        # Add text annotations for each value
        for i in range(true2init.shape[0]):
            for j in range(true2init.shape[1]):
                ax_init.text(j, i, f"{true2init[i, j]:.2f}", ha='center', va='center', color='white')

        # Plot true-to-pred
        ax_pred = axes[idx, 1]
        c2 = ax_pred.imshow(true2pred, aspect='auto', cmap=colormap, vmin=vmin, vmax=vmax)
        ax_pred.set_title(f"{metric} - True vs Pred")
        fig.colorbar(c2, ax=ax_pred)
        # Add text annotations for each value
        for i in range(true2pred.shape[0]):
            for j in range(true2pred.shape[1]):
                ax_pred.text(j, i, f"{true2pred[i, j]:.2f}", ha='center', va='center', color='white')

        # Plot true-to-true
        if idx == 1:
            ax_pred = axes[idx, 2]
            c2 = ax_pred.imshow(true2trueA, aspect='auto', cmap=colormap, vmin=vmin, vmax=vmax)
            ax_pred.set_title(f"{metric} - True vs True")
            fig.colorbar(c2, ax=ax_pred)
            # Add text annotations for each value
            for i in range(true2trueA.shape[0]):
                for j in range(true2trueA.shape[1]):
                    ax_pred.text(j, i, f"{true2trueA[i, j]:.2f}", ha='center', va='center', color='white')

        if idx == 2:
            ax_pred = axes[idx, 2]
            c2 = ax_pred.imshow(true2truew_element, aspect='auto', cmap=colormap, vmin=vmin, vmax=vmax)
            ax_pred.set_title(f"{metric} - True vs True")
            fig.colorbar(c2, ax=ax_pred)
            # Add text annotations for each value
            for i in range(true2truew_element.shape[0]):
                for j in range(true2truew_element.shape[1]):
                    ax_pred.text(j, i, f"{true2truew_element[i, j]:.2f}", ha='center', va='center', color='white')

        if idx == 3:
            ax_pred = axes[idx, 2]
            c2 = ax_pred.imshow(true2truew_vector, aspect='auto', cmap=colormap, vmin=vmin, vmax=vmax)
            ax_pred.set_title(f"{metric} - True vs True")
            fig.colorbar(c2, ax=ax_pred)
            # Add text annotations for each value
            for i in range(true2truew_vector.shape[0]):
                for j in range(true2truew_vector.shape[1]):
                    ax_pred.text(j, i, f"{true2truew_vector[i, j]:.2f}", ha='center', va='center', color='white')
        
        if idx == 4:
            ax_pred = axes[idx, 2]
            c2 = ax_pred.imshow(true2trueWin_element, aspect='auto', cmap=colormap, vmin=vmin, vmax=vmax)
            ax_pred.set_title(f"{metric} - True vs True")
            fig.colorbar(c2, ax=ax_pred)
            # Add text annotations for each value
            for i in range(true2trueWin_element.shape[0]):
                for j in range(true2trueWin_element.shape[1]):
                    ax_pred.text(j, i, f"{true2trueWin_element[i, j]:.2f}", ha='center', va='center', color='white')

        if idx == 5:
            ax_pred = axes[idx, 2]
            c2 = ax_pred.imshow(true2trueWin_vector, aspect='auto', cmap=colormap, vmin=vmin, vmax=vmax)
            ax_pred.set_title(f"{metric} - True vs True")
            fig.colorbar(c2, ax=ax_pred)
            # Add text annotations for each value
            for i in range(true2trueWin_vector.shape[0]):
                for j in range(true2trueWin_vector.shape[1]):
                    ax_pred.text(j, i, f"{true2trueWin_vector[i, j]:.2f}", ha='center', va='center', color='white')


        if idx == 6:
            ax_pred = axes[idx, 2]
            c2 = ax_pred.imshow(accuracy_true2tf, aspect='auto', cmap=colormap, vmin=np.min(accuracy_true2tf), vmax=np.max(accuracy_true2tf))
            ax_pred.set_title(f"{metric} - True vs True_fit")
            fig.colorbar(c2, ax=ax_pred)
            # Add text annotations for each value
            for i in range(accuracy_true2tf.shape[0]):
                for j in range(accuracy_true2tf.shape[1]):
                    ax_pred.text(j, i, f"{accuracy_true2tf[i, j]:.2f}", ha='center', va='center', color='white')

        if idx == 7:
            ax_pred = axes[idx, 2]
            c2 = ax_pred.imshow(precision_true2tf, aspect='auto', cmap=colormap, vmin=np.min(precision_true2tf), vmax=np.max(precision_true2tf))
            ax_pred.set_title(f"{metric} - True vs True_fit")
            fig.colorbar(c2, ax=ax_pred)
            # Add text annotations for each value
            for i in range(precision_true2tf.shape[0]):
                for j in range(precision_true2tf.shape[1]):
                    ax_pred.text(j, i, f"{precision_true2tf[i, j]:.2f}", ha='center', va='center', color='white')

        if idx == 8:
            ax_pred = axes[idx, 2]
            c2 = ax_pred.imshow(recall_true2tf, aspect='auto', cmap=colormap, vmin=np.min(recall_true2tf), vmax=np.max(recall_true2tf))
            ax_pred.set_title(f"{metric} - True vs True_fit")
            fig.colorbar(c2, ax=ax_pred)
            # Add text annotations for each value
            for i in range(recall_true2tf.shape[0]):
                for j in range(recall_true2tf.shape[1]):
                    ax_pred.text(j, i, f"{recall_true2tf[i, j]:.2f}", ha='center', va='center', color='white')

        if idx == 9:
            ax_pred = axes[idx, 2]
            c2 = ax_pred.imshow(f1_true2tf, aspect='auto', cmap=colormap, vmin=np.min(f1_true2tf), vmax=np.max(f1_true2tf))
            ax_pred.set_title(f"{metric} - True vs True_fit")
            fig.colorbar(c2, ax=ax_pred)
            # Add text annotations for each value
            for i in range(f1_true2tf.shape[0]):
                for j in range(f1_true2tf.shape[1]):
                    ax_pred.text(j, i, f"{f1_true2tf[i, j]:.2f}", ha='center', va='center', color='white')

        
                
    # Add overall figure title
    fig.suptitle("Metric Visualization", fontsize=16, y=1.02)

    # Display the plots
    plt.show()


def load_models(key, filename="metric_testing_model_data.json"):
    """Load all stored models for a given setting."""
    
    # Load JSON data
    if not os.path.exists(filename):
        print("No saved models found.")
        return None
    
    with open(filename, "r") as f:
        data = json.load(f)

    if key not in data:
        print(f"No models found for {key}")
        return None
    
    return data[key]

def train_and_store_model(N, K, D, dim_output, seed, testN = 3000, A_true=None, P_base_true=None, W_in_true=None, w_true=None, pi0_true=None, model_type=None, filename="id_metric_testing_model_data.json", n_init=3):
    """Train a model and store its parameters in JSON."""
    
    # Generate true model and data
    if model_type == "inputdriven":
        X_total, Y_total, states_seq_total, P_base_true, W_in_true, w_true, pi0_true, m_true = gen_true_param(N+testN, K, D, dim_output, seed, A_true=A_true, P_base_true=P_base_true, W_in_true=W_in_true, w_true=w_true, pi0_true=pi0_true, model_type=model_type)
        
    else:
        X_total, Y_total, states_seq_total, A_true, w_true, pi0_true, m_true = gen_true_param(N+testN, K, D, dim_output, seed, A_true=A_true, P_base_true=P_base_true, W_in_true=W_in_true, w_true=w_true, pi0_true=pi0_true, model_type=model_type)                                   
    
    X, X_test = X_total[:N], X_total[N:]
    Y, Y_test = Y_total[:N], Y_total[N:]
    true_states_seq = states_seq_total[N:]
    true_states_seq_fit = m_true.mostprob_states(X_test, Y_test).astype(int)

    # Train model
    if model_type == "inputdriven":

        P_base_init_all = []
        W_in_init_all = []
        w_init_all = []
        pi0_init_all = []
        init_states_seq_all = []

        
        lls_pred_all = []
        P_base_pred_all = []
        W_in_pred_all = []
        w_pred_all = []
        pi0_pred_all = []
        pred_states_seq_all = []
        
        for count in range(n_init):
            print(f"Init {count+1}")
            m = InputDrivenGLMHMM(N, K, D, dim_output, 1.0, list(np.arange(D + 1)))
            P_base_init = m.P_base.copy()
            W_in_init = m.W_in.copy()
            w_init = m.w.copy()
            pi0_init  = m.pi0.copy()
            init_states_seq = m.mostprob_states(X_test, Y_test).astype(int)

            lls_pred, P_base_pred, W_in_pred, w_pred, pi0_pred = m.fit(Y, X)
            pred_states_seq = m.mostprob_states(X_test, Y_test).astype(int)

            P_base_init_all.append(P_base_init)
            W_in_init_all.append(W_in_init)
            w_init_all.append(w_init)
            pi0_init_all.append(pi0_init)
            init_states_seq_all.append(init_states_seq)

            lls_pred_all.append(lls_pred)
            P_base_pred_all.append(P_base_pred)
            W_in_pred_all.append(W_in_pred)
            w_pred_all.append(w_pred)
            pi0_pred_all.append(pi0_pred)
            pred_states_seq_all.append(pred_states_seq)

 
        P_base_init_all = np.array(P_base_init_all)
        W_in_init_all = np.array(W_in_init_all)
        w_init_all = np.array(w_init_all)
        pi0_init_all = np.array(pi0_init_all)
        init_states_seq_all = np.array(init_states_seq_all)
        
        lls_pred_all = np.array(lls_pred_all)
        P_base_pred_all = np.array(P_base_pred_all)
        W_in_pred_all = np.array(W_in_pred_all)
        w_pred_all = np.array(w_pred_all)
        pi0_pred_all = np.array(pi0_pred_all)
        pred_states_seq_all = np.array(pred_states_seq_all)


        save_model_results(N, K, D, dim_output, X, Y, X_test, Y_test, None, P_base_true, W_in_true, w_true, pi0_true, true_states_seq, None, lls_pred_all, P_base_pred_all, W_in_pred_all, w_pred_all, pi0_pred_all, pred_states_seq_all,\
            None, P_base_init_all, W_in_init_all, w_init_all, pi0_init_all, init_states_seq_all, true_states_seq_fit, seed, model_type = model_type, filename=filename)

    else:   
        m = GLMHMM(N, K, D, dim_output, 1.0)

        A_init=m.transition_matrix.copy()
        w_init=m.w.copy()
        pi0_init  = m.pi0.copy()
        init_states_seq = m.mostprob_states(X_test, Y_test).astype(int)

        lls_pred, A_pred, w_pred, pi0_pred = m.fit(Y, X, np.copy(A_init), np.copy(w_init), pi0=np.copy(pi0_init), fit_init_states=True)
        pred_states_seq = m.mostprob_states(X_test, Y_test).astype(int)

        # Save the trained model's results
        save_model_results(N, K, D, dim_output, A_true, None, None, w_true, pi0_true, true_states_seq, A_pred, None, None, w_pred, pi0_pred, pred_states_seq,\
            A_init, None, None, w_init, pi0_init, init_states_seq, true_states_seq_fit, model_type = model_type, filename=filename)
   

def save_model_results(N, K, D, dim_output, X, Y, X_test, Y_test, A_true, P_base_true, W_in_true, w_true, pi0_true, true_states_seq, A_pred, lls_pred_all, P_base_pred_all, W_in_pred_all, w_pred_all, pi0_pred_all, pred_states_seq_all,\
    A_init, P_base_init_all, W_in_init_all, w_init_all, pi0_init_all, init_states_seq_all, true_states_seq_fit, seed, model_type=None, filename="metric_testing_model_data.json"):
    """Save trained model parameters to JSON, appending to existing settings if present."""
    
    # Convert NumPy arrays to lists for JSON storage
    if model_type == "inputdriven":
        model_data = {
            'X': X.tolist(),
            'Y': Y.tolist(),
            'X_test': X_test.tolist(),
            'Y_test': Y_test.tolist(),

            "P_base_true": P_base_true.tolist(),
            "W_in_true": W_in_true.tolist(),
            "w_true": w_true.tolist(),
            "pi0_true": pi0_true.tolist(),
            "true_states_seq": true_states_seq.tolist(),
            "true_states_seq_fit": true_states_seq_fit.tolist(),

            "P_base_init_all": P_base_init_all.tolist(),
            "W_in_init_all": W_in_init_all.tolist(),
            "w_init_all": w_init_all.tolist(),
            "pi0_init_all": pi0_init_all.tolist(),
            "init_states_seq_all": init_states_seq_all.tolist(),

            "P_base_pred_all": P_base_pred_all.tolist(),
            "W_in_pred_all": W_in_pred_all.tolist(),
            "w_pred_all": w_pred_all.tolist(),
            "pi0_pred_all": pi0_pred_all.tolist(),
            "pred_states_seq_all": pred_states_seq_all.tolist(),
            "lls_pred_all": lls_pred_all.tolist()
        }
    elif model_type is None:
        model_data = {
            "A_true": A_true.tolist(),
            "w_true": w_true.tolist(),
            "pi0_true": pi0_true.tolist(),
            "true_states_seq": true_states_seq.tolist(),
            "true_states_seq_fit": true_states_seq_fit.tolist(),

            "A_init": A_init.tolist(),
            "w_init": w_init_all.tolist(),
            "pi0_init": pi0_init_all.tolist(),
            "init_states_seq": init_states_seq_all.tolist(),

            "A_pred": A_pred.tolist(),
            "w_pred": w_pred_all.tolist(),
            "pi0_pred": pi0_pred_all.tolist(),
            "pred_states_seq": pred_states_seq_all.tolist(),
        }

    # Load existing JSON data
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Create a unique key based on (N, K, D, dim_output)
    key = f"N={N}_K={K}_D={D}_dim_output={dim_output}_seed={seed}"

    # Append new model data to existing key or create a new entry
    if key in data:
        data[key].append(model_data)  # Append new model
    else:
        data[key] = [model_data]  # First model for this setting

    # Save back to JSON
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved model under {key} in {filename}")



# set up a true glmhmm model and generate X, Y, and state seq from it
def gen_true_param(N, K, D, dim_output, seed, w_low = -1, w_high=1, A_true=None, P_base_true=None, W_in_true=None, w_true=None, pi0_true=None, model_type=None):
    div_pt = np.linspace(w_low, w_high, K+1)
    w_true = np.zeros((K, D + 1, dim_output))
    for d in range(D + 1):
        for k in range(K):
            w_true[k, d, :] = np.random.uniform(low=div_pt[k], high=div_pt[k+1], \
                    size = dim_output)

    # # bias <- 0s
    # w_true = np.pad(w_true, ((0, 0), (0, 1), (0, 0)), mode='constant')
    if model_type == "inputdriven":
        model_true = InputDrivenGLMHMM(N, K, D, dim_output, 1e-3, list(np.arange(2, D + 1))) #N, n_states, n_features, n_outputs
        model_true.w = w_true
        if P_base_true is not None:
            model_true.P_base = P_base_true
        if W_in_true is not None:
            model_true.W_in = W_in_true
        if w_true is not None:
            model_true.w = w_true
        if pi0_true is not None:
            model_true.pi0 = pi0_true

        P_base_true = model_true.P_base
        W_in_true = model_true.W_in
        w_true = model_true.w
        pi0_true = model_true.pi0

        X, Y, states = model_true.generate_data(N, seed)
        return X, Y, states, P_base_true, W_in_true, w_true, pi0_true, model_true
    
    else: 
        model_true = GLMHMM(N, K, D, dim_output, 1e-3) #N, n_states, n_features, n_outputs
        model_true.w = w_true
        if A_true is not None:
            model_true.transition_matrix = A_true
        if w_true is not None:
            model_true.w = w_true
        if pi0_true is not None:
            model_true.pi0 = pi0_true

        A_true = model_true.transition_matrix
        w_true = model_true.w
        pi0_true = model_true.pi0

        X, Y, states = model_true.generate_data(N)
        return X, Y, states, A_true, w_true, pi0_true, model_true

# Finding the optimal number of states using cross validation
def OptStateCV_traj(train_trajectories, traj_metric=None, model_name = "hmm", states_cands=[1,2,3,4,5], n_folds=6, inpts=None, num_init=3, obs_dist="gaussian", N_iters=100, verbose=False):
  print("\nCROSS VALIDATION\n")
  res_params = {}
  for state in states_cands:
     res_params[state] = {}
  if model_name == "hmm":
     pass
     ############TODO##############
     

  if inpts is not None:
    print("Use inputs")
    assert(len(train_trajectories) == len(inpts))
  fold_size = len(train_trajectories) // n_folds
  val_lls = np.zeros((len(states_cands), n_folds, num_init))  # array to store performance results
  # aris = np.zeros((len(states_cands), n_folds))
  states = np.ones((len(states_cands), n_folds, num_init, len(train_trajectories)))*(-1)
  for i, num_states in enumerate(states_cands):

    if model_name == "glmhmm":
      w_allfold = []
      A_allfold = []
      pi0_allfold = []

    print(f"---------------Number of states: {num_states}---------------")
    for fold in range(n_folds):
      print(f"####fold {fold+1}####")
      # Prepare the validation and training data
      val_start = fold * fold_size
      val_end = val_start + fold_size if fold < n_folds - 1 else len(train_trajectories)

      val_traj = train_trajectories[val_start:val_end]  # Validation set

      train_traj = list(train_trajectories[:val_start]) + list(train_trajectories[val_end:]) if val_start >0 else train_trajectories[val_end:] # Training set
      if inpts is not None:
        train_inpts = list(inpts[:val_start]) + list(inpts[val_end:]) if val_start>0 else inpts[val_end:]  # Training set
        # print("train_inpt, before", train_inpts.shape)
        # train_inpts = np.concatenate(train_inpts, axis=0)
        # print("train_inpt, after", train_inpts.shape)
        val_inpts = inpts[val_start:val_end]  # Validation set
        # val_inpts = np.concatenate(val_inpts, axis=0)

      # Apply transformation on observation if needed
      if traj_metric is not None:
          curv_train = traj_metric(train_traj)
          curv_val = traj_metric(val_traj)
      else:
          curv_train = np.array(train_traj)
          curv_val = np.array(val_traj)

      # fit hmm
      obs_dim=curv_train.shape[1] if curv_train.ndim > 1 else 1
      n_features = len(inpts[0]) if isinstance(inpts, list) else inpts.shape[1]

      if model_name == "glmhmm":
        ws = np.zeros((num_init, num_states, n_features + 1, obs_dim))
        As = np.zeros((num_init, num_states, num_states))
        pi0s = np.zeros((num_init, num_states))

      for init in range(num_init):
        if inpts is None:
          hmm = ssm.HMM(num_states, obs_dim, n_features,
                observations=obs_dist,
                transitions="inputdriven")
          model_lls = hmm.fit(curv_train, inputs=train_inpts, method="em", num_iters=N_iters, init_method="kmeans")
          train_ll = model_lls[-1]/len(curv_train)
          print("Train ll", train_ll)
        else:
          if model_name == "hmm":
            hmm = ssm.HMM(num_states, obs_dim, observations=obs_dist)
            model_lls = hmm.fit(curv_train, method="em", num_iters=N_iters, init_method="kmeans")
          else:
            assert model_name =='glmhmm'
            model = GLMHMM(len(curv_train), num_states, n_features, obs_dim, True) #N, n_states, n_features, n_outputs
            A=model.transition_matrix
            w=model.w
            lls,A,w,pi0 = model.fit(curv_train,train_inpts,A,w, fit_init_states=True)
            As[init] = A
            pi0s[init] = pi0
            ws[init] = w

            train_ll = last_non_nan(lls)/len(curv_train)
            print("Train ll", train_ll)
            

        # validate
        # loglikelihood
        if model_name == "hmm":
          if inpts is not None:
            performance = hmm.log_likelihood(curv_val, inputs=val_inpts)
          else:
            performance = hmm.log_likelihood(curv_val)
        else:
            assert model_name == "glmhmm"
        phi = np.zeros((len(curv_val), num_states))#(N, K)
        for k in range(num_states):
            thetak = model.dist_param(w[k], val_inpts, augment=True) # calculate theta
            for t in range(len(curv_val)):
                phi[t,k] = model.dist_pdf(curv_val[t], thetak[t], otherparamk=model.covariances[k]) # calculate phi
            performance,_ ,_ ,_  = model.forwardPass(curv_val,A,phi,pi0=pi0)
        val_lls[i, fold, init] = performance/len(curv_val)
        print(f"Val ll {performance/len(curv_val)}")
        # most likely states
        if inpts is not None:
          if model_name == "hmm":
            most_likely_states = hmm.most_likely_states(curv_val, input=val_inpts)
          else:
            assert model_name == "glmhmm"
            most_likely_states = model.mostprob_states(val_inpts, curv_val)
        else:
          most_likely_states = hmm.most_likely_states(curv_val)
        states[i, fold, init, :len(most_likely_states)] = most_likely_states
      # ari_list = []
      # for init_i in range(len(most_likely_states_list)):
      #   for init_j in range(init_i, len(most_likely_states_list)):
      #     ari_list.append(adjusted_rand_score(most_likely_states_list[init_i], most_likely_states_list[init_j]))
      # aris[i, fold] = np.mean(ari_list)

      w_allfold.append(ws)
      A_allfold.append(As)
      pi0_allfold.append(pi0s)


    if model == "hmm":
      ############TODO##############
      pass
    elif model == "glmhmm":
      res_params[num_states]['ws'] = np.array(w_allfold)
      res_params[num_states]["As"] = np.array(A_allfold)
      res_params[num_states]["pi0"] = np.array(pi0_allfold)

    # val_lls # num of states x num of folds
  if verbose:
      # Calculate mean and standard deviation across folds
      val_lls = val_lls.reshape(val_lls.shape[0], -1)

      mean_lls = np.mean(val_lls, axis=1)
      std_lls = np.std(val_lls, axis=1)
      # mean_ari = np.mean(aris, axis=1)
      # std_ari = np.std(aris, axis=1)
      # Plotting
      states_vis = 1+np.arange(val_lls.shape[0])  # Array of state indices

      plt.figure(figsize=(10, 6))
      plt.errorbar(states_vis, mean_lls, yerr=std_lls, fmt='o-', capsize=5, capthick=2, elinewidth=1)
      # plt.errorbar(states, mean_ari, yerr=std_ari, fmt='o-', capsize=5, capthick=2, elinewidth=1)

      plt.title('Validation Loglikelihood vs Number of States')
      plt.xlabel('Number of States')
      plt.ylabel('Average Log-Likelihood')
      plt.xticks(states_vis)  # Set x-ticks to state indices
      plt.grid()
      plt.show()
  return val_lls, states, res_params


# Retrain hmm with optimal number of states(2/3)
def RetrainHMM(train_trajectories, test_trajectories, opt_states, traj_metric, model='hmm', train_inpts=None, test_inpts=None, obs_dist='gaussian', N_iters=100):
    # prepare the data(curvature)
    curv_retrain = traj_metric(train_trajectories)
    obs_dim=curv_retrain.shape[1] if curv_retrain.ndim > 1 else 1
    if train_inpts is not None:
        print("Use inputs")
        train_inpts = np.concatenate(train_inpts, axis=0)
        test_inpts_org = test_inpts
        test_inpts = np.concatenate(test_inpts, axis=0)
        assert(len(curv_retrain) == len(train_inpts))
    if train_inpts is None:
        hmm = ssm.HMM(opt_states, obs_dim, observations=obs_dist)
        hmm_lls = hmm.fit(curv_retrain, method="em", num_iters=N_iters, init_method="kmeans")
    else:
        if model == "hmm":
            hmm = ssm.HMM(opt_states, obs_dim, train_inpts.shape[1],
                        observations=obs_dist,
                        transitions="inputdriven")
            hmm_lls = hmm.fit(curv_retrain, inputs=train_inpts, method="em", num_iters=N_iters, init_method="kmeans")
        else:
            assert model=='glmhmm'
            ##########TODO##########
    plt.plot(np.array(hmm_lls)/len(curv_retrain), label="EM")
    plt.xlabel("EM Iteration")
    plt.ylabel("Log Probability")
    plt.legend(loc="lower right")
    plt.show()

    curv_test = traj_metric(test_trajectories)
    print("Loglikelihood on test set", hmm.log_likelihood(curv_test, inputs=test_inpts)/len(curv_test))

    most_likely_states_list = []
    for i, curr_traj in enumerate(test_trajectories):
        curr_curv = traj_metric([curr_traj])
        if test_inpts is not None:
            most_likely_states_list.append(hmm.most_likely_states(curr_curv, input=test_inpts_org[i]))
        else:
            most_likely_states_list.append(hmm.most_likely_states(curr_curv))
    return hmm, most_likely_states_list