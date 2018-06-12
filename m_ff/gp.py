# -*- coding: utf-8 -*-
import warnings

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
# from scipy.optimize import fmin_l_bfgs_b

import logging

logger = logging.getLogger(__name__)


class GaussianProcess(object):
    """ Gaussian process class
    Class of GP regression of QM energies and forces
    
    Args:
        kernel (obj): A kernel object (typically a two or three body)
        noise (foat): The regularising noise level (typically named \sigma_n^2)
        optimizer (str): The kind of optimization of marginal likelihood (not implemented yet)

    Attributes:
        x_train (list): The configurations used for training
        alpha (array): The coefficients obtained during training
        L (array): The lower triangular matrix from cholesky decomposition of gram matrix
    """

    def __init__(self, kernel=None, noise=1e-10, optimizer=None, n_restarts_optimizer=0):

        self.kernel = kernel
        self.noise = noise

        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer

        self.x_train = None
        self.y_train = None

        self.log_marginal_likelihood_value = None

        self.L = None
        self.alpha = None

    def fit(self, x, y):
        """Fit a Gaussian process regression model.

        Args:
            x (list): training configurations
            y (np.ndarray): training forces

        """

        self.x_train = x
        self.y_train = np.reshape(y, (y.shape[0] * 3, 1))

        self.log_marginal_likelihood_value = self.log_marginal_likelihood(self.kernel.theta)

        # Precompute quantities required for predictions which are independent of actual query points
        K = self.kernel.calc_gram(self.x_train)
        K[np.diag_indices_from(K)] += self.noise

        try:
            self.L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError as exc:
            exc.args = "The kernel, {}, is not returning a positive definite matrix. Try gradually " + \
                       "increasing the 'noise' parameter of your GaussianProcessRegressor estimator. {}". \
                           format(self.kernel, exc.args)
            raise

        self.alpha = cho_solve((self.L, True), self.y_train)  # Line 3

    def predict(self, x, return_std=False):
        """Predict using the Gaussian process regression model

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True)

        Args:
            x (np.ndarray): Target configurations where the GP is evaluated
            return_std (bool): If True, the standard-deviation of the
                predictive distribution of the target configurations is
                returned along with the mean.

        Returns:
            y_mean (np.ndarray): Mean of predictive distribution at target configurations.
            y_std (np.ndarray): Standard deviation of predictive distribution at target
                configurations. Only returned when return_std is True.
        """

        # Predict based on GP posterior
        K_trans = self.kernel.calc(x, self.x_train)

        y_mean = K_trans.dot(self.alpha)  # Line 4 (y_mean = f_star)

        if return_std:

            # compute inverse K_inv of K based on its Cholesky  decomposition L and its inverse L_inv
            L_inv = solve_triangular(self.L.T, np.eye(self.L.shape[0]))
            K_inv = L_inv.dot(L_inv.T)

            # Compute variance of predictive distribution
            y_var = self.kernel.calc_diag(x)
            fit = np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans)
            y_var -= fit

            # Check if any of the variances is negative because of numerical issues. If yes: set the variance to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn("Predicted variances smaller than 0. Setting those variances to 0.")
                y_var[y_var_negative] = 0.0

            return y_mean.reshape(-1, 3), np.sqrt(y_var)
        else:
            return y_mean.reshape(-1, 3)

    def predict_energy(self, x, return_std=False):
        """Predict energies using the Gaussian process regression model

        This function evaluates the GP energies for a set of test configurations.

        Args:
            x (np.ndarray): Target configurations where the GP is evaluated

        Returns:
            y_mean (np.ndarray): Predicted energies
            y_std (np.ndarray): Predicted error on the energies
        """

        # Predict based on GP posterior
        K_trans = self.kernel.calc_ef(x, self.x_train)
        e_mean = K_trans.dot(self.alpha)  # Line 4 (y_mean = f_star)

        if return_std:
            # compute inverse K_inv of K based on its Cholesky decomposition L and its inverse L_inv
            L_inv = solve_triangular(self.L.T, np.eye(self.L.shape[0]))
            K_inv = L_inv.dot(L_inv.T)

            # Compute variance of predictive distribution
            e_var = self.kernel.calc_diag_e(x)
            fit = np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans)
            e_var -= fit

            # Check if any of the variances is negative because of numerical issues. If yes: set the variance to 0.
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

        Args:
            theta : array-like, shape = (n_kernel_params,) or None
                Kernel hyperparameters for which the log-marginal likelihood is
                evaluated. If None, the precomputed log_marginal_likelihood
                of ``self.kernel_.theta`` is returned.
            eval_gradient : bool, default: False
                If True, the gradient of the log-marginal likelihood with respect
                to the kernel hyperparameters at position theta is returned
                additionally. If True, theta must not be None.

        Returns:
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
            return self.log_marginal_likelihood_value

        # kernel = self.kernel_.clone_with_theta(theta)
        kernel = self.kernel
        kernel.theta = theta

        if eval_gradient:
            K, K_gradient = kernel.calc_gram(self.x_train, eval_gradient=True)
        else:
            K = kernel.calc_gram(self.x_train)

        K[np.diag_indices_from(K)] += self.noise

        try:
            L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train
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
            log_likelihood_gradient_dims = 0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def save(self, filename):
        """Dump the current GP model for later use

        Args:
            filename (str): name of the file where to save the GP

        Todo:
            * Need to decide the way to store a GP
        """

        output = [self.kernel.kernel_name,
                  self.noise,
                  self.optimizer,
                  self.n_restarts_optimizer,
                  self.alpha,
                  self.K,
                  self.x_train]

        np.save(filename, output)
        logger.info('Saved Gaussian process with name: {}'.format(filename))

    def load(self, filename):
        """Load a saved GP model

        Args:
            name (str): name of the file where the GP is saved

        Todo:
            * Need to decide the way to store a GP
        """

        # TODO: Proper initialisation of the kernel based on its name

        self.kernel.kernel_name, \
        self.noise, \
        self.optimizer, \
        self.n_restarts_optimizer, \
        self.alpha, \
        self.K, \
        self.x_train = np.load(filename)

        self.kernel = self.kernel

        logger.info('Loaded GP from file!')
