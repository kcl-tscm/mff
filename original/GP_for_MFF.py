import warnings

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import fmin_l_bfgs_b

np.set_printoptions(precision=3)


class GaussianProcess:
    """ GaussianProcess
    Class of GP regression of QM energies and forces
    
    Parameters
    ----------
    kernel: A kernel object, typically a two or three body
    noise: The regularising noise level (\sigma_n^2)
    optimizer: The kind of optimization of marginal likelihood (not implemented)
    
    Attributes
    ----------
    X_train_: The configurations used for training
    alpha_: The coefficients obtained during training
    L_: The lower triangular matrix from cholesky decomposition of gram matrix
    """

    # optimizers "fmin_l_bfgs_b"

    def __init__(self, kernel=None, noise=1e-10,
                 optimizer=None, n_restarts_optimizer=0):

        self.kernel = kernel
        self.noise = noise
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer

    def fit(self, X, y):
        """Fit Gaussian process regression model.
        Parameters
        ----------
        X : 
        y : 
        
        Returns
        -------
        self : returns an instance of self.
        """
        self.kernel_ = self.kernel

        self.X_train_ = X
        self.y_train_ = np.reshape(y, (y.shape[0] * 3, 1))

        if self.optimizer is not None:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta)

            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      self.kernel_.theta,
                                                      self.kernel_.bounds))]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_.calc_gram(self.X_train_)
        K[np.diag_indices_from(K)] += self.noise

        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'noise' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise

        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3
        self.K = K
        return self

    def predict(self, X, return_std=False):
        """Predict using the Gaussian process regression model
        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True)
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated
        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.
        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean
        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points
        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.
        """

        if not hasattr(self, "X_train_"):  # Unfitted; predict based on GP prior
            kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            if return_std:
                y_var = kernel.calc_diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

        else:  # Predict based on GP posterior
            K_trans = self.kernel_.calc(X, self.X_train_)

            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)

            if return_std:
                # compute inverse K_inv of K based on its Cholesky
                # decomposition L and its inverse L_inv
                L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
                K_inv = L_inv.dot(L_inv.T)
                # Compute variance of predictive distribution

                y_var = self.kernel_.calc_diag(X)
                fit = np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans)
                y_var -= fit

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0
                return np.reshape(y_mean, (int(y_mean.shape[0] / 3), 3)), np.sqrt(y_var)
            else:
                return np.reshape(y_mean, (int(y_mean.shape[0] / 3), 3))

    def predict_energy(self, X, return_std=False):
        """
        This function evaluates the GP energies at X.
    
        Returns
        -------
        y : array_like
        """

        if not hasattr(self, "X_train_"):  # Unfitted; predict based on GP prior
            kernel = self.kernel
            e_mean = np.zeros(X.shape[0])
            if return_std:
                y_var = kernel.calc_diag_e(X)
                return e_mean, np.sqrt(e_var)
            else:
                return e_mean

        else:  # Predict based on GP posterior
            K_trans = self.kernel_.calc_ef(X, self.X_train_)
            e_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)

            if return_std:
                # compute inverse K_inv of K based on its Cholesky
                # decomposition L and its inverse L_inv
                L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
                K_inv = L_inv.dot(L_inv.T)
                # Compute variance of predictive distribution
                e_var = self.kernel_.calc_diag_e(X)
                fit = np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans)
                e_var -= fit

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                e_var_negative = e_var < 0
                if np.any(e_var_negative):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    e_var[e_var_negative] = 0.0
                return e_mean, np.sqrt(e_var)
            else:
                return e_mean

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.
        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.
        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.
        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        # kernel = self.kernel_.clone_with_theta(theta)
        kernel = self.kernel
        kernel.theta = theta

        if eval_gradient:
            K, K_gradient = kernel.calc_gram(self.X_train_, eval_gradient=True)
        else:
            K = kernel.calc_gram(self.X_train_)

        K[np.diag_indices_from(K)] += self.noise
        try:
            L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) \
                if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        alpha = cho_solve((L, True), y_train)  # Line 3

        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions

        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            # Compute "0.5 * trace(tmp.dot(K_gradient))" without
            # constructing the full matrix tmp.dot(K_gradient) since only
            # its diagonal is required
            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            theta_opt, func_min, convergence_dict = \
                fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
            if convergence_dict["warnflag"] != 0:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                              " state: %s" % convergence_dict)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min

    def save(self, filename):

        output = [self.kernel_,
                  self.noise,
                  self.optimizer,
                  self.n_restarts_optimizer,
                  self.alpha_,
                  self.K,
                  self.X_train_]

        np.save(filename, output)
        print('Saved Gaussian process with name:', filename)

    def load(self, filename):
        self.kernel_, \
        self.noise, \
        self.optimizer, \
        self.n_restarts_optimizer, \
        self.alpha_, \
        self.K, \
        self.X_train_ = np.load(filename)

        print('Loaded GP from file')
