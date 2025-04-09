import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy import optimize
from scipy.special import logsumexp



class InputDrivenGLMHMM:
    def __init__(self, N, n_states, n_features, n_outputs, covar_epsilon, W_in_offidx = [], max_iter=100, em_dist="gaussian"):
        self.N = N
        self.n_states = n_states  # Number of hidden states (K)
        self.n_features = n_features + 1  # Features + bias term
        self.n_outputs = n_outputs
        self.max_iter = max_iter
        self.optimizer_tol = None #0.0001

        # Initialize
        if em_dist == "gaussian":
            self.pdf = multivariate_normal.pdf
            self.w = np.random.uniform(low=-1, high=1, size=(self.n_states, self.n_features, self.n_outputs))
            self.covariances = np.array([covar_epsilon * np.eye(n_outputs) for _ in range(n_states)])
        else:
            self.pdf = None

        A = np.random.gamma(1*np.ones((self.n_states, self.n_states)) + 5*np.identity(self.n_states),1)
        self.P_base = A/np.repeat(np.reshape(np.sum(A,axis=1),(1, self.n_states)),self.n_states,0).T
        self.W_in = np.random.uniform(low=-.8, high=.8, size=(self.n_states, self.n_features))
        # if len(W_in_offidx) > 0:
        #     self.W_in[:, W_in_offidx] = np.random.uniform(low=-0.01, high=0.01, size=(self.n_states, len(W_in_offidx)))
        self.pi0 = (1 / self.n_states) * np.ones(self.n_states)

        self.data_generate_param = None

    def transition_probabilities(self, u_t):
        if u_t.ndim == 1:
            u_t = u_t.reshape((-1, 1))

        log_P = np.log(self.P_base) + (self.W_in @ u_t).T
        P = np.exp(log_P - logsumexp(log_P, axis=1, keepdims=True))
        return P


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
        
    def fit(self, y, x, pi0=None, maxiter=250, tol=1e-3):
        if x.shape[1] != self.n_features:
            x = np.hstack([x, np.ones((x.shape[0], 1))])  # Add bias term
        
        self.lls = np.full(maxiter, np.nan)

        if pi0 is not None:
            self.pi0 = pi0

        # Compute initial emission probabilities
        phi = np.zeros((self.N, self.n_states))
        for k in range(self.n_states):
            theta_k = self.dist_param(self.w[k], x)
            for t in range(self.N):
                phi[t, k] = self.dist_pdf(y[t], theta_k[t], self.covariances[k])

        for n in range(maxiter):
            print(f"Iter {n+1}")

            # E Step: Forward-Backward
            ll, alpha, cs = self.forwardPass(y, x, phi)
            beta = self.backwardPass(y, x, phi, alpha, cs)
            gamma = np.multiply(alpha, beta)  # Posterior probabilities

            # Compute expected joint probabilities
            # xi = self.compute_xi(y, x, alpha, beta, cs)
            xi = np.zeros((self.N - 1, self.n_states, self.n_states))

            for t in range(self.N - 1):
                P_t = self.transition_probabilities(x[t])
                xi[t] = ((beta[t+1] * phi[t,:]) * np.reshape(alpha[t], (self.n_states, 1))) * P_t
                xi[t] /= cs[t + 1]

            # M Step: Update parameters
            self.pi0 = np.divide(gamma[0],sum(gamma[0]))
            self._updateTransitionParameters(x, xi)
            phi = self._updateEmissionParameters(y, x, gamma)

            # Check convergence
            self.lls[n] = ll
            if n > 5 and self.lls[n-5] + tol >= ll:
                break

        return self.lls, self.P_base, self.W_in, self.w, self.pi0

    def forwardPass(self, y, x, phi):
        alpha = np.zeros((self.N, self.n_states))
        cs = np.zeros(self.N)

        alpha[0] = np.multiply(phi[0,:],np.squeeze(self.pi0))

        cs[0] = np.sum(alpha[0])
        alpha[0] /= cs[0]

        for t in range(1, self.N):
            P_t = self.transition_probabilities(x[t])
            alpha[t] = np.multiply(phi[t,:], alpha[t-1] @ P_t)
            cs[t] = np.sum(alpha[t])
            alpha[t] /= cs[t]

        ll = np.sum(np.log(cs))
        return ll, alpha, cs

    def backwardPass(self, y, x, phi, alpha, cs):
        beta = np.zeros((self.N, self.n_states))
        beta[-1] = 1

        for t in range(self.N - 2, -1, -1):
            P_t = self.transition_probabilities(x[t + 1])
            beta[t] = (P_t @ np.multiply(beta[t + 1], phi[t+1,:])) / cs[t + 1]

        return beta

    def _updateEmissionParameters(self, y, x, gamma):
        phi = np.zeros((self.N,self.n_states))

        for k in range(self.n_states):
            self.w[k], phi[:, k] = self._glmfit(x, self.w[k], y, gammak=gamma[:, k])
            thetak = self.dist_param(self.w[k], x)
            residuals = y - thetak
            self.covariances[k] = np.cov(residuals.T)
        return phi
    

    def _updateTransitionParameters(self, x, xi):

        xis_n = np.reshape(np.sum(xi,axis=0),(self.n_states,self.n_states)) # sum_N xis
        xis_kn = np.reshape(np.sum(np.sum(xi,axis=0),axis=1),(self.n_states,1)) # sum_k sum_N xis
        self.P_base = xis_n/xis_kn

        def objective(W_flat):
            W = W_flat.reshape(self.n_states, self.n_features)
            log_likelihood = sum([np.sum(xi[t] * np.log(self.transition_probabilities(x[t]))) for t in range(self.N - 1)])
            return -log_likelihood/self.N

        def gradient(W_flat):
            W = W_flat.reshape(self.n_states, self.n_features)
            grad = np.zeros_like(W)
            for k in range(self.n_states):
                over_t = np.zeros((self.N - 1, self.n_features))
                for t in range(self.N-1):
                    over_t[t] = np.sum((xi[t, :, k].reshape(-1, 1) @ x[t].reshape(1, -1)) - (xi[t, :, k]*\
                        np.sum(self.transition_probabilities(x[t]), axis=1)).reshape(-1, 1) @ x[t].reshape(1, -1), axis=0)
                grad[k] = np.sum(over_t, axis=0)
            return -grad.flatten()
        # optimizer_state = self.optimizer_state if hasattr(self, "optimizer_state") else None
        # self.params = optimizer(objective, self.params, num_iters=num_iters,\
        #               state=optimizer_state, full_output=True, **kwargs)
        # result = minimize(objective, self.W_in.flatten(), jac=gradient, method="L-BFGS-B", options={'maxiter': 150})
        result = minimize(objective, self.W_in.flatten(), method="L-BFGS-B", jac=None, options={'maxiter': 150})

        self.W_in = result.x.reshape(self.n_states, self.n_features)

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
        ll_list = [gammak[i] * np.log(self.dist_pdf(y[i], thetak[i], otherparamk=otherparamk)) for i in range(self.N)]
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
        OptimizeResult = optimize.minimize(opt_log, w_flat, jac = "True", method = "L-BFGS-B", tol=self.optimizer_tol) # tol default see /usr/local/lib/python3.10/dist-packages/scipy/optimize/_lbfgsb_py.py

        wk = np.reshape(OptimizeResult.x,(self.n_features,self.n_outputs)) # reshape and update weights
        thetak = self.dist_param(wk, x) # calculate theta
        phi = self.dist_pdf(y, thetak, otherparamk=otherparamk) # calculate phi

        return wk, phi
    


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
        if X.shape[1] != self.n_features:
            X = np.hstack([X, np.ones((X.shape[0], 1))])  # Augment with bias term
        N = X.shape[0]  
        log_prob = np.zeros((N, self.n_states))  
        prev_state = np.zeros((N, self.n_states), dtype=int)

        log_prob[0] = np.log(self.pi0) + np.log(self._compute_likelihood(X[0], Y[0]))

        for t in range(1, N):
            likelihood = np.log(self._compute_likelihood(X[t], Y[t]))

            for k in range(self.n_states):
                transition_probs = log_prob[t - 1] + np.log(self.transition_probabilities(X[t])[:, k])
                best_prev_state = np.argmax(transition_probs)
                log_prob[t, k] = transition_probs[best_prev_state] + likelihood[k]
                prev_state[t, k] = best_prev_state

        states = np.zeros(N, dtype=int)
        states[-1] = np.argmax(log_prob[-1]) 

        for t in range(N - 2, -1, -1):
            states[t] = prev_state[t + 1, states[t + 1]]

        return states
    
    
    #----------------------------------------------------------------------#
    # Date generation

    def generate_data(self, n_samples, seed, X=None, numChange=0, p=3, method='sinusoidal'):

        assert method in ['autoregressive', 'sinusoidal']

        np.random.seed(seed)
        states = np.zeros(n_samples, dtype=int)
        states[0] = np.random.choice(self.n_states, p=self.pi0)
        
        X = np.zeros((n_samples, self.n_features))
        X[0] = np.random.randn(self.n_features)
        transition_probs = self.transition_probabilities(X[0])

        Y = np.zeros((n_samples, self.n_outputs))
        Y[0] = np.random.multivariate_normal(
            mean=self.dist_param(self.w[states[0]], X[0]),
            cov=self.covariances[states[0]]
        )

        if method == 'autoregressive':
            if self.data_generate_param is None:
                autoreg = np.random.dirichlet(np.ones(p), size=self.n_features) #D x p
                self.data_generate_param = autoreg
            else:
                autoreg = self.data_generate_param
            print('autoregressive:',autoreg)
            for t in range(1, n_samples):
                if t < p:
                    X[t] = (np.eye(self.n_features) @ X[t-1].reshape((-1, 1))).flatten()
                else:
                    X[t] = np.diag(autoreg @ X[t-p:t].reshape((p, self.n_features)))

                X[t] += np.random.normal(loc=0, scale=0.1, size=self.n_features) #scale=0.2
                
                transition_probs = self.transition_probabilities(X[t])
                states[t] = np.random.choice(self.n_states, p=transition_probs[states[t - 1]])
                Y[t] = np.random.multivariate_normal(
                    mean=self.dist_param(self.w[states[t]], X[t]),
                    cov=self.covariances[states[t]]
                )
        else:
            magnitude = 1.3 #0.7
            if self.data_generate_param is None:
                autoreg = np.random.dirichlet(np.ones(p), size=self.n_features) #D x p
                self.data_generate_param = autoreg
            else:
                autoreg = self.data_generate_param

            frequency = 1.7*np.pi*np.array([1/200, 1/400, 1/800]).reshape((-1, 1)) # length = p #2.7
            print('sinusoidal weight:',autoreg)
            for t in range(1, n_samples):
                X[t] = X[0] + magnitude * (autoreg @ np.sin(frequency * t)).flatten()
                transition_probs = self.transition_probabilities(X[t])
                states[t] = np.random.choice(self.n_states, p=transition_probs[states[t - 1]])
                Y[t] = np.random.multivariate_normal(
                    mean=self.dist_param(self.w[states[t]], X[t]),
                    cov=self.covariances[states[t]]
                )

        return X, Y, states