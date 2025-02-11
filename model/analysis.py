import numpy as np
import matplotlib.pyplot as plt
# from config import colors, actions
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment
from fitting import load_models

import numpy as np
import json
import os

def analyze_stored_models(N, K, D, dim_output, verbose=False, filename="metric_testing_model_data.json"):
    """Load and compare all stored models for a given setting."""

    key = f"N={N}_K={K}_D={D}_dim_output={dim_output}"
    stored_models = load_models(N, K, D, dim_output, filename=filename)
    
    if stored_models is None:
        return None

    num_models = len(stored_models)

    print(f"Analyzing {num_models} models for {key}")

    # Initialize storage for metrics
    storage_all = np.zeros((num_models, num_models, 3, 9))  # 2: init-0, pred-1 # (true, pred, true to init/pred/true, metric)

    # Loop through all pairs of models for comparison
    for true_i in range(num_models):
        for pred_i in range(num_models):
            # Extract the true model parameters
            true_states_seq = np.array(stored_models[true_i]["true_states_seq"])
            true_states_seq_fit = np.array(stored_models[true_i]["true_states_seq_fit"])
            A_true = np.array(stored_models[true_i]["A_true"])
            w_true = np.array(stored_models[true_i]["w_true"])

            # Extract the predicted model init parameters
            init_states_seq = np.array(stored_models[pred_i]["init_states_seq"])
            A_init = np.array(stored_models[pred_i]['A_init'])
            w_init = np.array(stored_models[pred_i]['w_init'])

            # Extract the predicted model parameters
            pred_states_seq = np.array(stored_models[pred_i]["pred_states_seq"])
            pred_states_seq_fit = np.array(stored_models[pred_i]["true_states_seq_fit"])
            A_pred = np.array(stored_models[pred_i]["A_pred"])
            w_pred = np.array(stored_models[pred_i]["w_pred"])

            # Extract the 2nd model parameters
            true_states_seq2 = np.array(stored_models[pred_i]["true_states_seq"])
            A_true2 = np.array(stored_models[pred_i]["A_true"])
            w_true2 = np.array(stored_models[pred_i]["w_true"])

            # Align state sequences using permutation
            perm_init = find_permutation(init_states_seq, true_states_seq)
            init_states_perm = np.array([perm_init[init_states_seq[idx]] for idx in range(len(init_states_seq))])
            A_perm_init = A_init[np.ix_(perm_init, perm_init)]
            w_perm_init = w_init[perm_init, :, :]

            perm = find_permutation(pred_states_seq, true_states_seq)
            pred_states_perm = np.array([perm[pred_states_seq[idx]] for idx in range(len(true_states_seq))])
            A_perm = A_pred[np.ix_(perm, perm)]
            w_perm = w_pred[perm, :, :]

            perm2 = find_permutation(true_states_seq2, true_states_seq)
            A_perm2 = A_true2[np.ix_(perm2, perm2)]
            w_perm2 = w_true2[perm2, :, :]

            # vis
            if verbose:
                timebin=200
                cmap = 'viridis'
                vmin, vmax = 0, np.max(true_states_seq)

                if true_i == pred_i:
                    plt.figure(figsize=(12, 4))
                    plt.title(f"true{true_i}, pred{pred_i}")
                    plt.plot(range(200), pred_states_perm[:200], 'red', label='pred')
                    plt.plot(range(200), true_states_seq[:200], 'blue', label = 'true')
                    plt.plot(range(200), true_states_seq_fit[:200], 'gray', label = "true_fit")
                    plt.legend()


                    plt.figure(figsize=(8, 6))

                    # First subplot
                    plt.subplot(311)
                    plt.imshow(true_states_seq[None, :], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                    plt.xlim(0, timebin)
                    plt.ylabel("$z_{\\mathrm{true}}$")
                    plt.yticks([])

                    # Second subplot
                    plt.subplot(312)
                    plt.imshow(true_states_seq_fit[None, :], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                    plt.xlim(0, timebin)
                    plt.ylabel("$z_{\\mathrm{true fit}}$")
                    plt.yticks([])

                    # Third subplot (New Addition)
                    plt.subplot(313)
                    plt.imshow(pred_states_perm[None, :], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)  # Replace `extra_data` with your third data source
                    plt.xlim(0, timebin)
                    plt.ylabel("$z_{\\mathrm{pred}}$")
                    plt.yticks([])
                    plt.xlabel("time")

                    plt.tight_layout()

                    plt.show()


            # Compute Metrics
            # Matrix comparison
            A_true2true_vector = matrix_comp(A_true, A_perm2, metric="vector")

            A_true2pred_element = matrix_comp(A_true, A_perm, metric="element") #0
            A_true2pred_vector = matrix_comp(A_true, A_perm, metric="vector") #1
            A_true2init_element = matrix_comp(A_true, A_perm_init, metric="element")
            A_true2init_vector = matrix_comp(A_true, A_perm_init, metric="vector")

            w_true2true_element = matrix_comp(w_true, w_perm2, metric="element")
            w_true2true_vector = matrix_comp(w_true, w_perm2, metric="vector")

            w_true2pred_element = matrix_comp(w_true, w_perm, metric="element") #2
            w_true2pred_vector = matrix_comp(w_true, w_perm, metric="vector") #3
            w_true2init_element = matrix_comp(w_true, w_perm_init, metric="element")
            w_true2init_vector = matrix_comp(w_true, w_perm_init, metric="vector")

            # State seq evaluation
            res_map_true2tf = evaluate_classification(true_states_seq, pred_states_seq_fit)
            accuracy_true2tf = res_map_true2tf['accuracy']
            precision_true2tf = res_map_true2tf['precision']
            recall_true2tf = res_map_true2tf['recall']
            f1_true2tf = res_map_true2tf['f1_score']

            res_map_true2pred = evaluate_classification(true_states_seq, pred_states_perm)
            accuracy_true2pred = res_map_true2pred['accuracy'] #4
            precision_true2pred = res_map_true2pred['precision'] #5
            recall_true2pred = res_map_true2pred['recall'] #6
            f1_true2pred = res_map_true2pred['f1_score'] #7

            res_map_true2init = evaluate_classification(true_states_seq, init_states_perm)
            accuracy_true2init = res_map_true2init['accuracy']
            precision_true2init = res_map_true2init['precision']
            recall_true2init = res_map_true2init['recall']
            f1_true2init = res_map_true2init['f1_score']

            # matching rate of state seq
            mat_rate_true2pred = calculate_match_rate(pred_states_perm, true_states_seq) #8
            mat_rate_true2init = calculate_match_rate(init_states_perm, true_states_seq)

            # Storage
            # true vs init
            storage_all[true_i, pred_i, 0] = np.array([A_true2init_element, A_true2init_vector, w_true2init_element, w_true2init_vector, \
                accuracy_true2init, precision_true2init, recall_true2init, f1_true2init, mat_rate_true2init])

            # true vs pred
            storage_all[true_i, pred_i, 1] = np.array([A_true2pred_element, A_true2pred_vector, w_true2pred_element, w_true2pred_vector, \
                accuracy_true2pred, precision_true2pred, recall_true2pred, f1_true2pred, mat_rate_true2pred])

            # true vs true (A only)
            storage_all[true_i, pred_i, 2, 1] = A_true2true_vector

            storage_all[true_i, pred_i, 2, 2] = w_true2true_element
            storage_all[true_i, pred_i, 2, 3] = w_true2true_vector

            # true vs true fit(state seq only) at place index 4, 5, 6, 7
            storage_all[true_i, pred_i, 2, 4] = accuracy_true2tf
            storage_all[true_i, pred_i, 2, 5] = precision_true2tf
            storage_all[true_i, pred_i, 2, 6] = recall_true2tf
            storage_all[true_i, pred_i, 2, 7] = f1_true2tf
            
    # Save analysis results
    analysis_file = "metric_testing_analysis_results.json"
    analysis_results = {key: storage_all.tolist()}

    with open(analysis_file, "w") as f:
        json.dump(analysis_results, f, indent=4)

    print(f"Analysis saved to {analysis_file}")
    return storage_all



# State matching using A or w
def permute_states(M,method='self-transitions',param='transitions',order=None,ix=None):

    '''
    Parameters
    ----------
    M : matrix of probabilities for input parameter (transitions, observations, or initial states)
    Methods ---
        self-transitions : permute states in order from highest to lowest self-transition value (works
             only with transition probabilities as inputs)
        order : permute states according to a given order
    param : specifies the input parameter
    order : optional, specifies the order of permuted states for method=order

    Returns
    -------
    M_perm : M permuted according to the specified method/order
    order : the order of the permuted states
    '''

    # check for valid method
    method_list = {'self-transitions','order','weight value'}
    if method not in method_list:
        raise Exception("Invalid method: {}. Must be one of {}".
            format(method, method_list))

    # sort according to transitions
    if method =='self-transitions':

        if param != 'transitions':
            raise Exception("Invalid parameter choice: self-transitions permutation method \
                            requires transition probabilities as parameter function input")
        diags = np.diagonal(M) # get diagonal values for sorting

        order = np.flip(np.argsort(diags))

        M_perm = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_perm[i,j] = M[order[i],order[j]]
     # sort according to given order
    if method == 'order':
        if param=='transitions':
            M_perm = np.zeros_like(M)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    M_perm[i,j] = M[order[i],order[j]]
        if param=='observations':
            M_perm = np.zeros_like(M)
            for i in range(M.shape[0]):
                M_perm[i,:] = M[order[i],:]
        if param=='weights':
            M_perm = np.zeros_like(M)
            for i in range(M.shape[0]):
                M_perm[i,:,:] = M[order[i],:,:]
        if param=='states':
            K = len(np.unique(M))
            M_perm = np.zeros_like(M)
            for i in range(K):
                M_perm[M==i] = order[i]
        if param=='pstates':
            M_perm = np.zeros_like(M)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    M_perm[i,j] = M[i,order[j]]

    # sort by the value of a particular weight
    if method == 'weight value':
        if ix is None:
            raise Exception("Index of weight ix must be specified for this method")

        order = np.flip(np.argsort(M[:,ix]))

        M_perm = np.zeros_like(M)
        for i in range(M.shape[0]):
            M_perm[i,:] = M[order[i],:]

    return M_perm, order.astype(int) # order: state[idx] go to idx place

# State matching using most probable state sequence
def compute_state_overlap(z1, z2, K1=None, K2=None):
    # assert z1.dtype == int and z2.dtype == int
    # assert z1.shape == z2.shape
    # assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    overlap = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap[k1, k2] = np.sum((z1 == k1) & (z2 == k2))
    return overlap

def find_permutation(z1, z2, K1=None, K2=None):
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape

    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm # state idx in z1 correspondes to state perm[idx] in z2


# pairwise ARI on different state sequences after state matching
# we want to constaint permutating state in ari calculation process

def evaluate_classification(y_true, y_pred):
    """
    Evaluates classification performance.

    Args:
        y_true (list or array-like): Ground truth (true labels).
        y_pred (list or array-like): Predicted labels.

    Returns:
        dict: A dictionary with accuracy, precision, recall, and F1-score.
    """
    assert len(y_true) == len(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def calculate_match_rate(z1, z2):
    """
    Calculate the match rate (fraction of correctly matched samples) between two clustering assignments,
    assuming the labels of the two arrays already match.

    Args:
        z1: array-like of shape (n_samples,)
        z2: array-like of shape (n_samples,)

    Returns:
        match_rate: Fraction of correctly matched samples.
    """
    assert len(z1) == len(z2), "The two arrays must have the same length"
    # Count the number of matching elements
    matches = np.sum(z1 == z2)
    # Calculate the fraction of matches
    match_rate = matches / len(z1)
    return match_rate


# Matrix comparison(w and A)
def matrix_comp(m_true, m_pred, metric="element"):
    metric_list = ['element', 'vector']
    if metric not in metric_list:
        raise Exception("Invalid comparison metric: {}. Must be one of {}".
            format(metric, metric_list))
    if metric == "element":
        arr_true = m_true.flatten()
        arr_pred = m_pred.flatten()
        assert(len(arr_true) == len(arr_pred))

        err_list = []
        for idx in range(len(arr_true)):
            x_true = arr_true[idx]
            x_pred = arr_pred[idx]
            err = np.abs((x_pred - x_true) / x_true)
            err_list.append(err)
        return np.mean(err_list)
    else:
        # Frobenius norm
        return np.linalg.norm(m_pred - m_true) / np.linalg.norm(m_true)


# def visWs(weights, labels=['wind_x', 'wind_y', 'odor', \
#                             'agent_angle_x', 'agent_angle_y', \
#                             'ego_course_direction_x', 'ego_course_direction_y']):
#     for i in range(weights.shape[0]):
#         plt.plot(range(weights.shape[1]), weights[i], color=colors[i], label = f"state {i+1}")
#     plt.xticks(ticks = range(weights.shape[1]), labels=labels,
#             rotation=90)
#     plt.plot(range(weights.shape[1]), np.zeros(weights.shape[1]), 'k--')
#     plt.legend()
#     plt.show()

# def visStateDist(feature, states, feature_name:str):
#     # Create a DataFrame to organize the data for plotting
#     data = pd.DataFrame({'curv': feature, 'states': states})

#     # Plot the distribution of `curv` values for each state using a boxplot
#     plt.figure(figsize=(10, 6))
#     # sns.boxplot(x='states', y='curv', data=data)
#     # Box plot for the overall distribution
#     sns.boxplot(x='states', y=feature_name, data=data, color="lightblue", width=0.6)

#     # # Overlay a swarm plot for individual data points
#     # sns.swarmplot(x='states', y='curv', data=data, color=".25", size=3)
#     plt.title(f'Distribution of {feature_name} Values for Each State')
#     plt.xlabel('State')
#     plt.ylabel(f'{feature_name} Value')
#     # plt.grid(True)
#     plt.show()

# def visColorAct(states):
#     vstack_actions = np.vstack(actions)
#     T = len(vstack_actions)
#     for tp in range(T - 1):
#         plt.plot([tp,0], vstack_actions[tp,1], '.', color = colors[states[tp]])
#     plt.xlabel("Step")
#     plt.ylabel("Turn")
#     plt.show()

###########TODO#############
#Map the state coloring back to the neural dynamics 