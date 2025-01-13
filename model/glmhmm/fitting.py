import numpy as np
import os
def CrossValidation(path, nprev, exig, num_latent, num_folds = 10, num_init = 3, verbose=False):
    

    # Model parameters
    N = len(data) - len(data)//num_folds # number of data/time points
    train_size = N
    test_size = len(data)//num_folds
    K = num_latent # number of latent states
    D = data.shape[1] - 2 # number of GLM inputs (regressors)
    if exig:
        C = 2 # number of observation classes
        prob = "bernoulli"
    else:
        C = 3 # number of observation classes
        prob = "multinomial"
        
    # store values for cross validation
    lls_train = np.zeros((num_init, num_folds))
    lls_test =  np.zeros((num_init, num_folds))
    ll0 =  np.zeros((num_init, num_folds))
    
    A_all = np.zeros((num_init, num_folds,K,K))
    w_all = np.zeros((num_init, num_folds,K,D,C))
    pi0_all = np.zeros((num_init, num_folds,K))

    # Set up the model
    model = GLMHMM(n=N,d=D,c=C,k=K,observations=prob, gaussianPrior=1) # set up a new GLM-HMM
    
    # Perform cross-validation
    fold_size = len(data) // num_folds
    for j in range(num_init):
        A_init,w_init,pi_init = model.generate_params() # initialize the model parameters
        for i in range(num_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size
            
            # Split data into train and test sets
            test_data = data[start_idx:end_idx]
            if end_idx < len(data):
                train_data = np.concatenate((data[:start_idx], data[end_idx:]), axis=0)
            else:
                train_data = data[:start_idx]
    
            X_train = train_data[:,1:-1]
            y_train = train_data[:,-1]
            X_test = test_data[:,1:-1]
            y_test = test_data[:,-1]
            probR = np.sum(y_train)/len(y_train)

            model.n = len(y_train)
            ll, A, w, pi0 = model.fit(y_train,X_train,A_init,w_init,pi0=pi_init, fit_init_states=True, sess=sess) # fit the model on trainset
            #print(np.linalg.norm(w-w_init))
            ll = find_last_non_nan_elements(ll.reshape(1, -1))
            lls_train[j, i] = ll[0]
            A_all[j,i] = A
            w_all[j,i] = w
            pi0_all[j,i] = pi0

            # testset
            #GLMHMM.n = test_size
            # convert inferred weights into observation probabilities for each state
            phi = np.zeros((len(X_test),K,C))
            for k in range(K):
                phi[:,k,:] = model.glm.compObs(X_test,w[k,:,:])

            # compute inferred log-likelihoods
            lls_test[j,i],_,_,_ = model.forwardPass(y_test,A,phi)
            ll0[j,i] = np.log(probR) * np.sum(y_test) + np.log(1 - probR) * (len(y_test) - np.sum(y_test)) # base probability
            
        if verbose:
            print('train ll', lls_train[j,i])
            print('test ll',lls_test[j,i])
            print('Init %s complete' %(j+1))
    
    return lls_train, lls_test, ll0, A_all, w_all, pi0_all, train_size, test_size