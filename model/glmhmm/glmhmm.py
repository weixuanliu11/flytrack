import numpy as np
from scipy.stats import multivariate_normal
from scipy import optimize

class GLMHMM:
    def __init__(self, N, n_states, n_features, n_outputs, max_iter=100, em_dist="gaussian"):
        self.N = N
        self.n_states = n_states # K
        self.n_features = n_features + 1 # D
        self.n_outputs = n_outputs # dim of y
        self.max_iter = max_iter

        # Initialize
        if em_dist == "gaussian":
            self.pdf = multivariate_normal.pdf
            self.w = np.random.randn(self.n_states, self.n_features - 1, self.n_outputs)
            self.w = np.pad(self.w, ((0, 0), (0, 1), (0, 0)), mode='constant')
            self.covariances = np.array([np.eye(n_outputs) for _ in range(n_states)])
        else:
            self.pdf = None
        
        self.transition_matrix = np.full((self.n_states, self.n_states), 1 / self.n_states) # uniformly distributed


    def dist_param(self, wk, x):
        """
        Human designed formula of f where theta_t = f(w_k, x_t)
        wk has shape (D, output dim)
        x has shape (N, D)
        Return distribution parameter over time - thetak of shape (N, output dim)
        """
        pre_act = x @ wk
        return np.tanh(pre_act)
        thetak = np.zeros((x.shape[0], self.n_outputs))
        for i in range(x.shape[0]):
            thetak[i] = np.tanh(x[i] @ wk)
        return thetak 

    def neglogli(self, wk, x, y, gammak, otherparamk=None, reshape_weights=False):
        """
        Compute the negative log-likelihood of the observation given each state.
        wk has shape (D, output dim)
        x has shape (N, D)
        thetak has shape (N, output dim)
        y has shape (N, output dim)
        gammak has shape (N, )
        Return a number -sum_t (gamma_t * log(p(y_t | theta_t)))
        """
        if reshape_weights:
          wk = wk.reshape((self.n_features - 1, self.n_outputs))
          wk = np.vstack((wk, np.zeros((1, self.n_outputs))))

        thetak = self.dist_param(wk, x)
        ll_list = [gammak[i] * np.log(self.dist_pdf(y[i], thetak[i], otherparamk=otherparamk)) for i in range(self.n_states)]
        ll = np.sum(ll_list)
        return -ll

    def dist_pdf(self, y, thetak, otherparamk=None):
        """
        Calculate the pdf of a distribution, given y and parameters
        return f(y) of shape (N, )
        y has shape (N, output dim)
        thetak has shape (N, output dim)
        """
        thetakt = []
        if self.pdf is not None:
            if y.ndim == 1:
                return self.pdf(y, mean=thetak, cov=otherparamk)
            
            for t in range(self.N):
                thetakt.append(self.pdf(y[t], mean=thetak[t], cov=otherparamk))
            return np.array(thetakt)
        else:
          raise Exception("No distribution function defined")
        return 0

    def _compute_likelihood(self, xt, yt):
        ll = []
        for k in range(self.n_states):
            wk = self.w[k]
            thetakt = self.dist_param(wk, xt)
            ll.append(self.dist_pdf(yt, thetakt, otherparamk=self.covariances[k]))
        return np.array(ll)
        

    def predict(self, X, Y):
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, self.n_states))
        prev_states = np.zeros((n_samples, self.n_states), dtype=int)

        # Augment X with intercept column for prediction
        X_augmented = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept column

        log_probs[0] = np.log(self._compute_likelihood(X_augmented[0], Y[0])) + np.log(1 / self.n_states)

        for t in range(1, n_samples):
            for j in range(self.n_states):
                log_probs[t, j] = np.max(log_probs[t - 1] + np.log(self.transition_matrix[:, j])) + \
                                  np.log(self._compute_likelihood(X_augmented[t], Y[t])[j])
                prev_states[t, j] = np.argmax(log_probs[t - 1] + np.log(self.transition_matrix[:, j]))

        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(log_probs[-1])
        for t in range(n_samples - 2, -1, -1):
            states[t] = prev_states[t + 1, states[t + 1]]

        return states

    def generate_data(self, n_samples):
        """
        Generate synthetic data given the model parameters.

        Returns:
        - X: Input features.
        - Y: Observations.
        - states: True hidden states.
        """
        X = np.random.randn(n_samples, self.n_features-1)
        states = np.zeros(n_samples, dtype=int)
        Y = np.zeros((n_samples, self.n_outputs))

        # Generate initial state and observation
        states[0] = np.random.choice(self.n_states)
        X_augmented = np.hstack([X, np.ones((X.shape[0], 1))])  # Add intercept column
        Y[0] = np.random.multivariate_normal(
            mean=X_augmented[0] @ self.w[states[0]],
            cov=self.covariances[states[0]]
        )

        # Generate subsequent states and observations
        for t in range(1, n_samples):
            states[t] = np.random.choice(self.n_states, p=self.transition_matrix[states[t - 1]])
            Y[t] = np.random.multivariate_normal(
                mean=X_augmented[t] @ self.w[states[t]],
                cov=self.covariances[states[t]]
            )

        return X, Y, states


    def _updateTransitions(self,y,alpha,beta,cs,A,phi):

        '''
        Updates transition probabilities as part of the M-step of the EM algorithm.
        Currently only functional for stationary transitions (GLM on transitions not supported)
        Uses closed form updates as described in Bishop Ch. 13

        Parameters
        ----------
        y : nx1 vector of observations
        alpha : nx1 vector of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        beta : nx1 vector of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        cs : nx1 vector of the forward marginal likelihoods
        A : kxk matrix of transition probabilities
        phi : kxc or nxkxc matrix of emission probabilities

        Returns
        -------
        A_new : kxk matrix of updated transition probabilities

        '''
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

    def _updateObservations(self,y,x,w,gammas):

        '''
        Updates emissions probabilities as part of the M-step of the EM algorithm.
        For stationary observations, see the HMM class
        Uses gradient descent to find optimal update of weights

        Parameters
        ----------
        y : nx1 vector of observations
        gammas : nxk matrix of the posterior probabilities of the latent states

        Returns
        -------
        kxc matrix of updated emissions probabilities
        '''

        # reshape y from vector of indices to one-hot encoded array for matrix operations in glm.fit
        if y.ndim == 1:
          y = y.reshape((-1, 1))

        self.phi = np.zeros((self.N,self.n_states))

        for zk in np.arange(self.n_states):
            self.w[zk], self.phi[:,zk] = self._glmfit(x,w[zk],y,self.covariances[zk], 
                                                      #compHess=self.hessian,
                                                      gammak=gammas[:,zk],
                                                      #gaussianPrior=self.gaussianPrior
                                                      )

            thetak = self.dist_param(self.w[zk], x)
            residuals = y - thetak
            self.covariances[zk] = np.cov(residuals.T)

        return self.w, self.phi

    def _glmfit(self,x,wk,y,otherparamk=None, compHess=False,gammak=None,gaussianPrior=0):
        """
        Use gradient descent to optimize weights
        wk has shape (D, output dim)
        x has shape (N, D)
        Return: phi is of shape (N, )
        """

        w_flat = np.ndarray.flatten(wk[:-1,:]) # flatten weights for optimization
        opt_log = lambda w: self.neglogli(w, x, y, gammak, otherparamk=otherparamk, reshape_weights=True) # calculate log likelihood

        OptimizeResult = optimize.minimize(opt_log, w_flat, jac = "True", method = "L-BFGS-B")
      
        wk = np.vstack((np.reshape(OptimizeResult.x,(self.n_features-1,self.n_outputs)), np.zeros((1, self.n_outputs)))) # reshape and update weights
        thetak = self.dist_param(wk, x) # calculate theta
        phi = self.dist_pdf(y, thetak, otherparamk=otherparamk) # calculate phi

        return wk, phi

    def _updateInitStates(self,gammas):
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

    def fit(self,y,x,A,w,pi0=None,fit_init_states=False,maxiter=250,tol=1e-3,sess=None,B=1):
        '''
        Parameters
        ----------
        y : nx1 vector of observations
        A : initial kxk matrix of transition probabilities
        phi : initial kxc or nxkxc matrix of emission probabilities
        pi0 : initial kx1 vector of state probabilities for t=1.
        fit_init_states : boolean, determines if EM will including fitting pi
        maxiter : int. The maximum number of iterations of EM to allow. The default is 250.
        tol : float. The tolerance value for the loglikelihood to allow early stopping of EM. The default is 1e-3.
        sessions : an optional vector of the first and last indices of different sessions in the data (for
        separate computations of the E step; first and last entries should be 0 and n, respectively)
        B : an optional temperature parameter used when fitting via direct annealing EM (DAEM; see Ueda and Nakano 1998)
        Returns
        -------
        lls : vector of loglikelihoods for each step of EM, size maxiter
        A : fitted kxk matrix of transition probabilities
        w : fitted kxdxc omatrix of weights
        pi0 : fitted kx1 vector of state probabilities for t= (only different from initial value of fit_init_states=True)
        '''
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
            # phi[:,k] = self.dist_pdf(y, thetak, otherparamk=self.covariances[k]) # calculate phi

        if sess is None:
            sess = np.array([0,self.N]) # equivalent to saying the entire data set has one session

        for n in range(maxiter):

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


                ll += ll_s
                alpha[sess[s]:sess[s+1]] = alpha_s
                cs[sess[s]:sess[s+1]] = cs_s
                self.pStates[sess[s]:sess[s+1]] = pBack_s ** B
                beta[sess[s]:sess[s+1]] = beta_s
                self.states[sess[s]:sess[s+1]] = zhatBack_s


            self.lls[n] = ll

            # M STEP
            A,w,phi,pi0 = self._updateParams(y,x,self.pStates,beta,alpha,cs,A,phi,w,fit_init_states = fit_init_states)


            # CHECK FOR CONVERGENCE
            self.lls[n] = ll
            if  n > 5 and self.lls[n-5] + tol >= ll: # break early if tolerance is reached
                break

        self.transition_matrix,self.w,self.phi,self.pi0 = A,w,phi,pi0

        return self.lls,self.transition_matrix,self.w,self.pi0


    def forwardPass(self,y,A,phi,pi0=None):

        '''
        Computes forward pass of Expectation Maximization (EM) algorithm; first half of E-step.

        Parameters
        ----------
        y : nx1 vector of observations
        A : kxk matrix of transition probabilities
        phi : nxkxc matrix of emission probabilities

        Returns
        -------
        ll : float, marginal log-likelihood of the data p(y)
        alpha : nx1 vector of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        cs : nx1 vector of the forward marginal likelihoods

        '''

        alpha = np.zeros((y.shape[0],self.n_states)) # forward probabilities p(z_t | y_1:t)
        alpha_prior = np.zeros_like(alpha) # prior probabilities p(z_t | y_1:t-1)
        cs = np.zeros(y.shape[0]) # forward marginal likelihoods

        if not np.any(pi0):
            pi0 = np.ones(self.n_states)/self.n_states

        # first time bin
        pxz = np.multiply(phi[0,:],np.squeeze(pi0)) # weight t=0 observation probabilities by initial state probabilities
        cs[0] = np.sum(pxz) # normalizer

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

        '''
        Computes backward pass of Expectation Maximization (EM) algorithm; second half of "E-step".

        Parameters
        ----------
        y : nx1 vector of observations
        A : kxk matrix of transition probabilities
        phi : nxkxc matrix of emission probabilities
        alpha : nx1 vector of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        cs : nx1 vector of the forward marginal likelihoods

        Returns
        -------
        pBack : nxk matrix of the posterior probabilities of the latent states
        beta : nx1 vector of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        zhatBack : nx1 vector of the most probable state at each time point

        '''

        beta = np.zeros((y.shape[0],self.n_states))

        # last time bin
        beta[-1] = 1 # take beta(z_N) = 1

        # backward pass for remaining time bins
        for i in np.arange(y.shape[0]-2,-1,-1):
            beta_prior = np.multiply(beta[i+1],phi[i+1,:]) # propogate uncertainty backward
            beta[i] = (A@beta_prior)/cs[i+1]

        pBack = np.multiply(alpha,beta) # posterior after backward pass -> alpha_hat(z_n)*beta_hat(z_n)
        zhatBack = np.argmax(pBack,axis=1) # decode from likelihoods only

        assert np.round(sum(pBack[0]),5) == 1, "Sum of posterior state probabilities does not equal 1"

        return pBack,beta,zhatBack