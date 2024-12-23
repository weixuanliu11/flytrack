import numpy as np
import matplotlib.pyplot as plt
from config import colors, actions
import seaborn as sns
import pandas as pd

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

    return M_perm, order.astype(int)

def visWs(weights, labels=['wind_x', 'wind_y', 'odor', \
                            'agent_angle_x', 'agent_angle_y', \
                            'ego_course_direction_x', 'ego_course_direction_y']):
    for i in range(weights.shape[0]):
        plt.plot(range(weights.shape[1]), weights[i], color=colors[i], label = f"state {i+1}")
    plt.xticks(ticks = range(weights.shape[1]), labels=labels,
            rotation=90)
    plt.plot(range(weights.shape[1]), np.zeros(weights.shape[1]), 'k--')
    plt.legend()
    plt.show()

def visStateDist(feature, states, feature_name:str):
    # Create a DataFrame to organize the data for plotting
    data = pd.DataFrame({'curv': feature, 'states': states})

    # Plot the distribution of `curv` values for each state using a boxplot
    plt.figure(figsize=(10, 6))
    # sns.boxplot(x='states', y='curv', data=data)
    # Box plot for the overall distribution
    sns.boxplot(x='states', y=feature_name, data=data, color="lightblue", width=0.6)

    # # Overlay a swarm plot for individual data points
    # sns.swarmplot(x='states', y='curv', data=data, color=".25", size=3)
    plt.title(f'Distribution of {feature_name} Values for Each State')
    plt.xlabel('State')
    plt.ylabel(f'{feature_name} Value')
    # plt.grid(True)
    plt.show()

def visColorAct(states):
    vstack_actions = np.vstack(actions)
    T = len(vstack_actions)
    for tp in range(T - 1):
        plt.plot([tp,0], vstack_actions[tp,1], '.', color = colors[states[tp]])
    plt.xlabel("Step")
    plt.ylabel("Turn")
    plt.show()

###########TODO#############
#Map the state coloring back to the neural dynamics 