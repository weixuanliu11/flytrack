from sklearn.metrics import adjusted_rand_score
from numpy import np
import model.ssm_package.ssm as ssm
import matplotlib.pyplot as plt
from glmhmm import GLMHMM

# Finding the optimal number of states using cross validation
def OptStateCV_traj(train_trajectories, traj_metric=None, model = "hmm", states_cands=[1,2,3,4,5], n_folds=6, inpts=None, num_init=3, obs_dist="gaussian", N_iters=100, verbose=False):
  res_params = {}
  for state in states_cands:
     res_params[state] = {}
  if model == "hmm":
     pass
     ############TODO##############
     

  if inpts is not None:
    print("Use inputs")
    assert(len(train_trajectories) == len(inpts))
  fold_size = len(train_trajectories) // n_folds
  val_lls = np.zeros((len(states_cands), n_folds, num_init))  # array to store performance results
  aris = np.zeros((len(states_cands), n_folds))
  for i, num_states in enumerate(states_cands):
    if model == "glmhmm":
      ws = np.zeros((n_folds, num_init, num_states, self.n_features - 1, self.n_outputs))
      As = np.zeros((n_folds, num_init, num_states, num_states))
      pi0s = np.zeros(())
    print(f"---------------Number of states: {num_states}---------------")
    for fold in range(n_folds):
      print(f"---------------fold {fold+1}---------------")
      # Prepare the validation and training data
      val_start = fold * fold_size
      val_end = val_start + fold_size if fold < n_folds - 1 else len(train_trajectories)

      val_traj = train_trajectories[val_start:val_end]  # Validation set
      train_traj = train_trajectories[:val_start] + train_trajectories[val_end:]  # Training set
      if inpts is not None:
        train_inpts = inpts[:val_start] + inpts[val_end:]  # Training set
        train_inpts = np.concatenate(train_inpts, axis=0)
        val_inpts = inpts[val_start:val_end]  # Validation set
        val_inpts = np.concatenate(val_inpts, axis=0)

      # Apply transformation on observation if needed
      if traj_metric is not None:
          curv_train = traj_metric(train_traj)
          curv_val = traj_metric(val_traj)
      else:
          curv_train = train_traj
          curv_val = val_traj

      # fit hmm
      obs_dim=curv_train.shape[1] if curv_train.ndim > 1 else 1
      most_likely_states_list = []
      for init in range(num_init):
        if inpts is not None:
          hmm = ssm.HMM(num_states, obs_dim, len(inpts[0][0]),
                observations=obs_dist,
                transitions="inputdriven")
          model_lls = hmm.fit(curv_train, inputs=train_inpts, method="em", num_iters=N_iters, init_method="kmeans")
          train_ll = model_lls[-1]/len(curv_train)
          print("Train ll", train_ll)
        else:
          if model == "hmm":
            hmm = ssm.HMM(num_states, obs_dim, observations=obs_dist)
            model_lls = hmm.fit(curv_train, method="em", num_iters=N_iters, init_method="kmeans")
          else:
            assert model=='glmhmm'
            ##########TODO###########
            model = GLMHMM(len(curv_train), num_states, len(inpts[0][0]), obs_dim) #N, n_states, n_features, n_outputs
            lls,A,w,pi0 = model.fit(curv_train,train_inpts,A,w, fit_init_states=True)
            As
            

        # validate
        # loglikelihood
        if inpts is not None:
          performance = hmm.log_likelihood(curv_val, inputs=val_inpts)
        else:
          performance = hmm.log_likelihood(curv_val)
        val_lls[i, fold, init] = performance/len(curv_val)
        print(f"Val ll {performance/len(curv_val)}")
        # most likely states
        if inpts is not None:
          most_likely_states = hmm.most_likely_states(curv_val, input=val_inpts)
        else:
          most_likely_states = hmm.most_likely_states(curv_val)
        most_likely_states_list.append(most_likely_states)
      ari_list = []
      for init_i in range(len(most_likely_states_list)):
        for init_j in range(init_i, len(most_likely_states_list)):
          ari_list.append(adjusted_rand_score(most_likely_states_list[init_i], most_likely_states_list[init_j]))
      aris[i, fold] = np.mean(ari_list)

    if model == "hmm":
      ############TODO##############
      pass
    elif model == "glmhmm":
      res_params[num_states]['ws'] = ws
      res_params[num_states]["As"] = As
      res_params[num_states]["pi0"] = pi0s

    # val_lls # num of states x num of folds
  if verbose:
      # Calculate mean and standard deviation across folds
      val_lls = val_lls.reshape(val_lls.shape[0], -1)

      mean_lls = np.mean(val_lls, axis=1)
      std_lls = np.std(val_lls, axis=1)
      mean_ari = np.mean(aris, axis=1)
      std_ari = np.std(aris, axis=1)
      # Plotting
      states = 1+np.arange(val_lls.shape[0])  # Array of state indices

      plt.figure(figsize=(10, 6))
      plt.errorbar(states, mean_lls, yerr=std_lls, fmt='o-', capsize=5, capthick=2, elinewidth=1)
      plt.errorbar(states, mean_ari, yerr=std_ari, fmt='o-', capsize=5, capthick=2, elinewidth=1)

      plt.title('Validation Loglikelihood vs Number of States')
      plt.xlabel('Number of States')
      plt.ylabel('Average Log-Likelihood')
      plt.xticks(states)  # Set x-ticks to state indices
      plt.grid()
      plt.show()
  return val_lls, aris, res_params


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