import numpy as np
from scipy.stats import multivariate_normal
from scipy import optimize
import time
from multiprocessing import Pool
 
class GLMHMM:
    def __init__(self, N, n_states, n_features, n_outputs, covar_epsilon, max_iter=100, em_dist="gaussian", A_dist="dirichlet"):
        """
        Initializes the GLMHMM class.

        Args:
            N (int): Number of data samples.
            n_states (int): Number of hidden states (K).
            n_features (int): Number of input features (D), excluding the bias term.
            n_outputs (int): Dimensionality of the output.
            max_iter (int): Maximum number of iterations for fitting.
            em_dist (str): Type of emission distribution ("gaussian" by default).

        Attributes:
            transition_matrix (ndarray): KxK matrix of transition probabilities.
            w (ndarray): KxDx n_outputs matrix of weights for the emission distributions.
            covariances (ndarray): Kx n_outputs x n_outputs covariance matrices for Gaussian emissions.
            pdf (callable): Probability density function for Gaussian emissions.
        """
        self.N = N
        self.n_states = n_states # K
        self.n_features = n_features + 1 # D
        self.n_outputs = n_outputs
        self.max_iter = max_iter
        self.optimizer_tol = None

        # Initialize
        if em_dist == "gaussian":
            self.pdf = multivariate_normal.pdf
            # weights ~ uniform(-1, 1)
            # bias <- 1
            w = np.random.uniform(low = -1, high = 1, size = (self.n_states, self.n_features - 1, self.n_outputs))
            self.w = np.pad(w, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=1)
            self.covariances = np.array([covar_epsilon*np.eye(n_outputs) for _ in range(n_states)])
        else:
            self.pdf = None
        if A_dist == "uniform":
            self.transition_matrix = np.full((self.n_states, self.n_states), 1 / self.n_states) # uniformly distributed
        elif A_dist == "dirichlet":
            A = np.random.gamma(1*np.ones((self.n_states, self.n_states)) + 5*np.identity(self.n_states),1)
            self.transition_matrix = A/np.repeat(np.reshape(np.sum(A,axis=1),(1, self.n_states)),self.n_states,0).T
        self.pi0 = (1/self.n_states) * np.ones(self.n_states)
        
    def dist_pdf(self, y, thetak, otherparamk=None):
        """
        Evaluates the probability density of observations for a given state.

        Args:
            y (ndarray): Observations, shape (N, output_dim).
            thetak (ndarray): Distribution parameters (e.g., mean), shape (N, output_dim).
            otherparamk (optional): Covariance matrix or other distribution-specific parameters.

        Returns:
            ndarray: Probability density values for each time step, shape (N,).
        """
        y = np.array(y)

        thetakt = []
        if self.pdf is not None:
            if y.ndim == 1:
                return self.pdf(y, mean=thetak, cov=otherparamk)
            
            for t in range(self.N):
                thetakt.append(self.pdf(y[t], mean=thetak[t], cov=otherparamk))
            return np.array(thetakt)
        else:
          raise Exception("No distribution function defined")


    def dist_param(self, wk, x, augment=False):
        """
        Computes the distribution parameters (e.g., the mean) for a given state and input.

        Args:
            wk (ndarray): Weights for a specific state, shape (D, output_dim).
            x (ndarray): Input data, shape (N, D).

        Returns:
            ndarray: Distribution parameters (e.g., mean) over time, shape (N, output_dim).
        """
        x = np.array(x)
        
        if augment:
            x = np.hstack([x, np.ones((x.shape[0], 1))])
        
        # pre_act = x @ wk 
        pre_act = (np.tanh(x @ wk) + 1 )/ 2 # action = (np.tanh(action) + 1)/2; line 1430 in tamagotchi.env
        return pre_act
        
    
    #--------------------------------------------------------------------------------#
    #EM Algorithm
    def fit(self,y,x,A,w,pi0=None,fit_init_states=False,maxiter=250,tol=1e-3,sess=None,B=1):
        """
        Fits the GLMHMM to the data using the EM algorithm.

        Args:
            y (ndarray): Observations, shape (N, output_dim).
            x (ndarray): Input features, shape (N, D-1).
            A (ndarray): Initial transition matrix, shape (K, K).
            w (ndarray): Initial weights, shape (K, D, output_dim).
            pi0 (optional): Initial state probabilities, shape (K,).
            fit_init_states (bool): Whether to optimize initial state probabilities.
            maxiter (int): Maximum number of EM iterations.
            tol (float): Convergence tolerance for the log-likelihood.
            sess (optional): Session boundaries for separate EM computation.
            B (float): Temperature parameter for annealing.

        Returns:
            tuple:
                - Log-likelihood values for each iteration.
                - Fitted transition matrix, weights, and initial probabilities.
        """
        x = np.array(x)
        y = np.array(y)

        x = np.hstack([x, np.ones((x.shape[0], 1))])

        self.lls = np.empty(maxiter)
        self.lls[:] = np.nan

        # store variables
        self.pi0 = pi0
        
        phi = np.zeros((self.N, self.n_states))#(N, K)
        for k in range(self.n_states):
            thetak = self.dist_param(w[k], x) # calculate theta
            for t in range(self.N):
                phi[t,k] = self.dist_pdf(y[t], thetak[t], otherparamk=self.covariances[k]) # calculate phi
                # print('theta[t]', thetak[t])
                # print('y[t]', y[t])
                # print('phi[t,k]', phi[t,k])
        # print("phi", phi)
        if sess is None:
            sess = np.array([0,self.N]) # equivalent to saying the entire data set has one session

        for n in range(maxiter):
            print(f"Iter{n+1}")

            if n == maxiter - 1:
                print("Reach the max number of EM iterations")

            # E STEP
            alpha = np.zeros((self.N,self.n_states))
            beta = np.zeros_like(alpha)
            cs = np.zeros((self.N))
            self.pStates = np.zeros_like(alpha)
            self.states = np.zeros_like(cs)
            ll = 0

            for s in range(len(sess)-1): # compute E step separately over each session or day of data
                ll_s,alpha_s,_,cs_s = self.forwardPass(y[sess[s]:sess[s+1]],A,phi[sess[s]:sess[s+1],:],pi0=pi0)
                pBack_s,beta_s,zhatBack_s = self.backwardPass(y[sess[s]:sess[s+1]],A,phi[sess[s]:sess[s+1],:],alpha_s,cs_s)
                # NOTE sess is buggy - here pass in data of window lengths - but in backwardPass, prior is calc'd by shape of N, which is the full data

                ll += ll_s
                alpha[sess[s]:sess[s+1]] = alpha_s
                cs[sess[s]:sess[s+1]] = cs_s
                self.pStates[sess[s]:sess[s+1]] = pBack_s ** B
                beta[sess[s]:sess[s+1]] = beta_s
                self.states[sess[s]:sess[s+1]] = zhatBack_s


            self.lls[n] = ll

            # M STEP
            start = time.time()
            A,w,phi,pi0 = self._updateParams(y,x,self.pStates,beta,alpha,cs,A,phi,w,fit_init_states = fit_init_states)
            print(f"Update params time: {time.time()-start}", flush=True)
            # print("M step")
            # print('A', A)
            # print('w', w)

            # CHECK FOR CONVERGENCE
            self.lls[n] = ll
            if  n > 5 and self.lls[n-5] + tol >= ll: # break early if tolerance is reached
                break

        self.transition_matrix,self.w,self.phi,self.pi0 = A,w,phi,pi0

        return self.lls,self.transition_matrix,self.w,self.pi0
    
    # E Step
    def forwardPass(self,y,A,phi,pi0=None):
        """
        Computes forward probabilities and log-likelihood during the E-step.

        Args:
            y (ndarray): Observations, shape (N, output_dim).
            A (ndarray): Transition matrix, shape (K, K).
            phi (ndarray): Emission probabilities, shape (N, K).
            pi0 (optional): Initial state probabilities, shape (K,).

        Returns:
            tuple:
                - Log-likelihood of the observations.
                - Forward probabilities (alpha), shape (N, K).
                - Forward marginal likelihoods (cs), shape (N,).
        """

        alpha = np.zeros((y.shape[0],self.n_states)) # forward probabilities p(z_t | y_1:t)
        alpha_prior = np.zeros_like(alpha) # prior probabilities p(z_t | y_1:t-1)
        cs = np.zeros(y.shape[0]) # forward marginal likelihoods

        if not np.any(pi0):
            pi0 = np.ones(self.n_states)/self.n_states

        # first time bin
        pxz = np.multiply(phi[0,:],np.squeeze(pi0)) # weight t=0 observation probabilities by initial state probabilities
        cs[0] = np.sum(pxz) # normalizer
        # cs[0] += 1e-9

        alpha[0] = pxz/cs[0] # conditional p(z_1 | y_1)
        alpha_prior[0] = 1/self.n_states # conditional p(z_0 | y_0)

        # forward pass for remaining time bins 
        for i in np.arange(1,y.shape[0]):
            alpha_prior[i] = alpha[i-1]@A # propogate uncertainty forward
            pxz = np.multiply(phi[i,:],alpha_prior[i]) # joint P(y_1:t,z_t)
            cs[i] = np.sum(pxz) # conditional p(y_t | y_1:t-1)
            alpha[i] = pxz/cs[i] # conditional p(z_t | y_1:t)
        ll = np.sum(np.log(cs))

        return ll,alpha,alpha_prior,cs


    def backwardPass(self,y,A,phi,alpha,cs):
        """
        Computes backward probabilities and posterior probabilities during the E-step.

        Args:
            y (ndarray): Observations, shape (N, output_dim).
            A (ndarray): Transition matrix, shape (K, K).
            phi (ndarray): Emission probabilities, shape (N, K).
            alpha (ndarray): Forward probabilities, shape (N, K).
            cs (ndarray): Scaling factors from forward pass, shape (N,).

        Returns:
            tuple:
                - Posterior probabilities, shape (N, K).
                - Backward probabilities (beta), shape (N, K).
                - Most probable states, shape (N,).
        """

        beta = np.zeros((y.shape[0],self.n_states))

        # last time bin
        beta[-1] = 1 # take beta(z_N) = 1

        # backward pass for remaining time bins
        for i in np.arange(self.N-2,-1,-1):
            beta_prior = np.multiply(beta[i+1],phi[i+1,:]) # propogate uncertainty backward
            # beta[i] = ((A@beta_prior)+1e-9)/cs[i+1]
            beta[i] = ((A@beta_prior))/cs[i+1]
        pBack = np.multiply(alpha,beta) # posterior after backward pass -> alpha_hat(z_n)*beta_hat(z_n)
        zhatBack = np.argmax(pBack,axis=1) # decode from likelihoods only

        # if np.round(sum(pBack[0]),5) == 1:
        #     print("Sum of posterior state probabilities does not equal 1")
        # else:
        #     print("equals 1") 

        return pBack,beta,zhatBack
    
    # M Step
    def _updateTransitions(self,y,alpha,beta,cs,A,phi):
        """
        Updates the transition probabilities during the M-step of the EM algorithm.

        Args:
            y (ndarray): Observations, shape (N, output_dim).
            alpha (ndarray): Forward probabilities, shape (N, K).
            beta (ndarray): Backward probabilities, shape (N, K).
            cs (ndarray): Scaling factors for forward probabilities, shape (N,).
            A (ndarray): Current transition matrix, shape (K, K).
            phi (ndarray): Emission probabilities, shape (N, K).

        Returns:
            ndarray: Updated transition matrix, shape (K, K).
        """

        # compute xis, the joint posterior distribution of two successive latent variables p(z_{t-1},z_t |Y,theta_old)
        xis = np.zeros((self.N-1,self.n_states,self.n_states))
        for i in np.arange(0,self.N-1):
            beta_phi = beta[i+1,:] * phi[i,:]
            alpha_reshaped = np.reshape(alpha[i,:],(self.n_states,1))
            xis[i,:,:] = ((beta_phi * alpha_reshaped) * A)/cs[i+1]

        # reshape and sum xis to obtain new transition matrix
        xis_n = np.reshape(np.sum(xis,axis=0),(self.n_states,self.n_states)) # sum_N xis
        xis_kn = np.reshape(np.sum(np.sum(xis,axis=0),axis=1),(self.n_states,1)) # sum_k sum_N xis
        A_new = xis_n/xis_kn

        return A_new

    def neglogli(self, wk, x, y, gammak, otherparamk=None, reshape_weights=False):
        """
        Computes the negative log-likelihood of observations for a specific state.

        Args:
            wk (ndarray): Weights for the current state, shape (D, output_dim).
            x (ndarray): Input data, shape (N, D).
            y (ndarray): Observations, shape (N, output_dim).
            gammak (ndarray): Posterior probabilities for the state, shape (N,).
            otherparamk (optional): Additional parameters for the distribution.
            reshape_weights (bool): If True, reshape weights to match dimensions.

        Returns:
            float: Negative log-likelihood for the state -- -sum_t (gamma_t * log(p(y_t | theta_t))). 
        """
        if reshape_weights:
            wk = wk.reshape((self.n_features, self.n_outputs))

        thetak = self.dist_param(wk, x)
        ll_list = [gammak[i] * np.log(self.dist_pdf(y[i], thetak[i], otherparamk=otherparamk) + 1e-10) for i in range(self.N)]
        ll = np.sum(ll_list)
        return -ll
    
    def _glmfit(self,x,wk,y,otherparamk=None, compHess=False,gammak=None,gaussianPrior=0):
        """
        Fits the GLM weights using gradient descent (L-BFGS-B).

        Args:
            x (ndarray): Input features, shape (N, D).
            wk (ndarray): Initial weights, shape (D, output_dim).
            y (ndarray): Observations, shape (N, output_dim).
            otherparamk (optional): Additional parameters for the distribution.
            compHess (bool): Compute the Hessian matrix (default=False).
            gammak (ndarray): Posterior probabilities for the state, shape (N,).
            gaussianPrior (float): Gaussian prior regularization parameter.

        Returns:
            tuple:
                - Optimized weights, shape (D, output_dim).
                - Emission probabilities, shape (N,).
        """

        w_flat = np.ndarray.flatten(wk) # flatten weights for optimization
        opt_log = lambda w: self.neglogli(w, x, y, gammak, otherparamk=otherparamk, reshape_weights=True) # calculate log likelihood

        # simplefilter(action='ignore', category=FutureWarning) # ignore FutureWarning generated by scipy
        print('Start optimize', flush=True)
        # start = time.time()
        # this is the slowest part of the code
        OptimizeResult = optimize.minimize(opt_log, w_flat, jac = "True", method = "L-BFGS-B", tol=self.optimizer_tol) # tol default see /usr/local/lib/python3.10/dist-packages/scipy/optimize/_lbfgsb_py.py

        wk = np.reshape(OptimizeResult.x,(self.n_features,self.n_outputs)) # reshape and update weights
        thetak = self.dist_param(wk, x) # calculate theta
        phi = self.dist_pdf(y, thetak, otherparamk=otherparamk) # calculate phi

        return wk, phi
    
    @staticmethod
    def fit_glm_for_state(zk, x, y, w, gammas, covariances, _glmfit, dist_param):
            w_zk, phi_zk = _glmfit(x, w[zk], y, covariances[zk], gammak=gammas[:, zk])
            thetak = dist_param(w_zk, x)
            residuals = y - thetak
            cov_zk = np.cov(residuals.T)
            return zk, w_zk, phi_zk, cov_zk
        
    def _updateObservations(self,y,x,w,gammas):
        """
        Updates the emission parameters (weights and covariances) during the M-step.

        Args:
            y (ndarray): Observations, shape (N, output_dim).
            x (ndarray): Input features, shape (N, D).
            w (ndarray): Current weights, shape (K, D, output_dim).
            gammas (ndarray): Posterior probabilities for the states, shape (N, K).

        Returns:
            tuple:
                - Updated weights, shape (K, D, output_dim).
                - Updated emission probabilities, shape (N, K).
        """

        # reshape y from vector of indices to one-hot encoded array for matrix operations in glm.fit
        if y.ndim == 1:
          y = y.reshape((-1, 1))

        self.phi = np.zeros((self.N,self.n_states))

        # for zk in np.arange(self.n_states):
        #     # print('zk', zk)
        #     # TODO: parallelize this loop - the optimization for each state is the slowest step
        #     self.w[zk], self.phi[:,zk] = self._glmfit(x,w[zk],y,self.covariances[zk], 
        #                                               gammak=gammas[:,zk],
        #                                               )

        #     thetak = self.dist_param(self.w[zk], x)
        #     residuals = y - thetak
        #     self.covariances[zk] = np.cov(residuals.T)
        
        with Pool() as pool:
            results = pool.starmap(self.fit_glm_for_state, [(zk, x, y, w, gammas, self.covariances, self._glmfit, self.dist_param) for zk in range(self.n_states)])

        for zk, w_zk, phi_zk, cov_zk in results:
            self.w[zk] = w_zk
            self.phi[:, zk] = phi_zk
            self.covariances[zk] = cov_zk
    
        return self.w, self.phi

    def _updateInitStates(self,gammas):
        """
        Updates the initial state probabilities during the M-step.

        Args:
            gammas (ndarray): Posterior probabilities for the states, shape (N, K).

        Returns:
            ndarray: Updated initial state probabilities, shape (K,).
        """
        return np.divide(gammas[0],sum(gammas[0])) # new initial latent state probabilities


    def _updateParams(self,y,x,gammas,beta,alpha,cs,A,phi,w,fit_init_states = False):
        '''
        Computes the updated parameters as part of the M-step of the EM algorithm.

        '''

        self.transition_matrix = self._updateTransitions(y,alpha,beta,cs,A,phi)

        self.w, self.phi = self._updateObservations(y,x,w,gammas)

        if fit_init_states:
            self.pi0 = self._updateInitStates(gammas)

        return self.transition_matrix, self.w, self.phi, self.pi0
    

    #----------------------------------------------------------------------#
    # Viterbi Decoding

    def _compute_likelihood(self, xt, yt):
        """
        Computes the likelihood of observations for all hidden states.

        Args:
            xt (ndarray): Input data for a single time step, shape (1, D).
            yt (ndarray): Observations for a single time step, shape (1, output_dim).

        Returns:
            ndarray: Likelihood values for all states, shape (K,).
        """

        ll = []
        for k in range(self.n_states):
            wk = self.w[k]
            thetakt = self.dist_param(wk, xt)
            ll.append(self.dist_pdf(yt, thetakt, otherparamk=self.covariances[k]))
        return np.array(ll)
        

    def mostprob_states(self, X, Y):
        """
        Decodes the most likely sequence of states using the Viterbi algorithm.

        Args:
            X (ndarray): Input data, shape (N, D).
            Y (ndarray): Observations, shape (N, output_dim).

        Returns:
            ndarray: Most likely sequence of states, shape (N,).
        """
        # X = np.array(X)
        # Y = np.array(Y)

        # n_samples = X.shape[0]
        # log_probs = np.zeros((n_samples, self.n_states))
        # prev_states = np.zeros((n_samples, self.n_states), dtype=int)

        # # Augment X with intercept column for prediction
        # X_augmented = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept column

        # log_probs[0] = np.log(self._compute_likelihood(X_augmented[0], Y[0])) + np.log(1 / self.n_states)

        # for t in range(1, n_samples):
        #     for j in range(self.n_states):
        #         log_probs[t, j] = np.max(log_probs[t - 1] + np.log(self.transition_matrix[:, j]) + \
        #                           np.log(self._compute_likelihood(X_augmented[t], Y[t])[j]))
        #         prev_states[t, j] = np.argmax(log_probs[t - 1] + np.log(self.transition_matrix[:, j]) + \
        #                           np.log(self._compute_likelihood(X_augmented[t], Y[t])[j]))        

        # states = np.zeros(n_samples, dtype=int)
        # states[-1] = np.argmax(log_probs[-1])
        # for t in range(n_samples - 2, -1, -1):
        #     states[t] = prev_states[t + 1, states[t + 1]]

        # return states

        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Augment X with bias term
        N = X.shape[0]  # Number of time steps
        log_prob = np.zeros((N, self.n_states))  # Log probabilities for each state
        prev_state = np.zeros((N, self.n_states), dtype=int)  # Backtrack pointers

        # Compute log-likelihood for first time step
        log_prob[0] = np.log(self.pi0 + 1e-10) + np.log(self._compute_likelihood(X[0], Y[0]) + 1e-10)

        # Viterbi forward pass
        for t in range(1, N):
            likelihood = np.log(self._compute_likelihood(X[t], Y[t]) + 1e-10)

            for k in range(self.n_states):
                transition_probs = log_prob[t - 1] + np.log(self.transition_matrix[:, k] + 1e-10)
                best_prev_state = np.argmax(transition_probs)
                log_prob[t, k] = transition_probs[best_prev_state] + likelihood[k]
                prev_state[t, k] = best_prev_state

        # Backtrack to find most probable sequence
        states = np.zeros(N, dtype=int)
        states[-1] = np.argmax(log_prob[-1])  # Start from the most probable last state

        for t in range(N - 2, -1, -1):
            states[t] = prev_state[t + 1, states[t + 1]]

        return states

    

    #----------------------------------------------------------------------#
    # Date generation

    def generate_data(self, n_samples, X=None):
        """
        Generates synthetic data using the model's parameters.

        Args:
            n_samples (int): Number of data samples to generate.

        Returns:
            tuple: 
                - X (ndarray): Generated input features, shape (N, D-1).
                - Y (ndarray): Generated observations, shape (N, output_dim).
                - states (ndarray): True hidden states, shape (N,).
        """
        if X is None:
            X = np.random.randn(n_samples, self.n_features-1)
        states = np.zeros(n_samples, dtype=int)
        Y = np.zeros((n_samples, self.n_outputs))

        # Generate initial state and observation
        states[0] = np.random.choice(self.n_states, p=self.pi0)
        X_augmented = np.hstack([X, np.ones((X.shape[0], 1))])  # Add intercept column
        Y[0] = np.random.multivariate_normal(
            mean=self.dist_param(self.w[states[0]], X_augmented[0]),
            cov=self.covariances[states[0]]
        )

        # Generate subsequent states and observations
        for t in range(1, n_samples):
            states[t] = np.random.choice(self.n_states, p=self.transition_matrix[states[t - 1]])
            Y[t] = np.random.multivariate_normal(
                mean=self.dist_param(self.w[states[t]], X_augmented[t]),
                cov=self.covariances[states[t]]
            )

        return X, Y, states