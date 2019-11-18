# -*- coding: utf-8 -*-

import logging

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import fmin_l_bfgs_b

from mff import interpolation, kernels

logger = logging.getLogger(__name__)


class GaussianProcess(object):
    """ Gaussian process class
    Class of GP regression of QM energies and forces

    Args:
        kernel (obj): A kernel object (typically a two or three body)
        noise (float): The regularising noise level (typically named \sigma_n^2)
        optimizer (str): The kind of optimization of marginal likelihood (not implemented yet)

    Attributes:
        X_train_ (list): The configurations used for training
        alpha_ (array): The coefficients obtained during training
        L_ (array): The lower triangular matrix from cholesky decomposition of gram matrix
        K (array): The kernel gram matrix
    """

    # optimizers "fmin_l_bfgs_b"

    def __init__(self, kernel=None, noise=1e-10,
                 optimizer=None, n_restarts_optimizer=0):

        self.kernel = kernel
        self.noise = noise
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.fitted = [None, None]

    def calc_gram_ff(self, X):
        """Calculate the force-force kernel gram matrix

        Args:
            X (list): list of N training configurations, which are M x 5 matrices

        Returns:
            K (matrix): The force-force gram matrix, has dimensions 3N x 3N
        """

        self.kernel_ = self.kernel
        self.X_train_ = X
        K = self.kernel_.calc_gram(self.X_train_, self.ncores)
        return K

    def calc_gram_ee(self, X):
        """Calculate the force-force kernel gram matrix

        Args:
            X (list): list of N training configurations, which are M x 5 matrices

        Returns:
            K (matrix): The energy energy gram matrix, has dimensions N x N
        """

        self.kernel_ = self.kernel
        self.X_train_ = X
        K = self.kernel_.calc_gram_e(self.X_train_, self.ncores)
        return K

    def fit(self, X, y, ncores=1):
        """Fit a Gaussian process regression model on training forces

        Args:
            X (list): training configurations
            y (np.ndarray): training forces
            ncores (int): number of CPU workers to use, default is 1

        """
        self.kernel_ = self.kernel
        self.X_train_ = X
        self.y_train_ = np.reshape(y, (y.shape[0] * 3, 1))

        # if self.optimizer is not None:
        #     # Choose hyperparameters based on maximizing the log-marginal
        #     # likelihood (potentially starting from several initial values)
        #     def obj_func(theta, eval_gradient=True):
        #         if eval_gradient:
        #             lml, grad = self.log_marginal_likelihood(
        #                 theta, eval_gradient=True)
        #             return -lml, -grad
        #         else:
        #             return -self.log_marginal_likelihood(theta)

        #     # First optimize starting from theta specified in kernel_
        #     optima = [(self._constrained_optimization(obj_func,
        #                                               self.kernel_.theta,
        #                                               self.kernel_.bounds))]

        #     # Additional runs are performed from log-uniform chosen initial
        #     # theta
        #     if self.n_restarts_optimizer > 0:
        #         if not np.isfinite(self.kernel_.bounds).all():
        #             raise ValueError(
        #                 "Multiple optimizer restarts (n_restarts_optimizer>0) "
        #                 "requires that all bounds are finite.")
        #         bounds = self.kernel_.bounds
        #         for iteration in range(self.n_restarts_optimizer):
        #             theta_initial = \
        #                 self._rng.uniform(bounds[:, 0], bounds[:, 1])
        #             optima.append(
        #                 self._constrained_optimization(obj_func, theta_initial,
        #                                                bounds))
        #     # Select result from run with minimal (negative) log-marginal
        #     # likelihood
        #     lml_values = list(map(itemgetter(1), optima))
        #     self.kernel_.theta = optima[np.argmin(lml_values)][0]
        #     self.log_marginal_likelihood_value_ = -np.min(lml_values)
        # else:
        #     self.log_marginal_likelihood_value_ = \
        #         self.log_marginal_likelihood(self.kernel_.theta)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_.calc_gram(self.X_train_, ncores)
        K[np.diag_indices_from(K)] += self.noise

        try:  # Use Cholesky decomposition to build the lower triangular matrix
            self.L_ = cholesky(K, lower=True)
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'noise' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise

        # Calculate the alpha weights using the Cholesky method
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)
        self.K = K
        self.energy_alpha_ = None
        self.energy_K = None
        self.X_glob_train_ = None
        self.fitted[0] = 'force'
        self.n_train = len(self.y_train_) // 3

        return self

    def fit_force_and_energy(self, X, y_force, X_glob, y_energy, ncores=1):
        """Fit a Gaussian process regression model using forces and energies

        Args:
            X (list of arrays): training configurations
            y_force (np.ndarray): training forces
            X_glob (list of lists of arrays): list of grouped training configurations
            y_energy (np.ndarray): training total energies
            ncores (int): number of CPU workers to use, default is 1

        """
        self.kernel_ = self.kernel
        self.X_train_ = X
        self.X_glob_train_ = X_glob
        self.y_train_ = np.reshape(y_force, (y_force.shape[0] * 3, 1))
        self.y_train_energy_ = np.reshape(y_energy, (y_energy.shape[0], 1))

        if self.optimizer is not None:
            logger.warning(
                "Optimizer not yet implemented for force-energy training")
            '''
            
            # TODO
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
            '''
        else:
            pass
        '''
             self.log_marginal_likelihood_value_ = \
                 self.log_marginal_likelihood(self.kernel_.theta)'''

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K_ff = self.kernel_.calc_gram(self.X_train_, ncores)
        K_ff[np.diag_indices_from(K_ff)] += self.noise

        K_ee = self.kernel_.calc_gram_e(self.X_glob_train_, ncores)
        K_ee[np.diag_indices_from(K_ee)] += self.noise

        K_ef = self.kernel_.calc_gram_ef(
            self.X_train_, self.X_glob_train_, ncores)

        K = np.zeros((y_force.shape[0] * 3 + y_energy.shape[0],
                      y_force.shape[0] * 3 + y_energy.shape[0]))
        K[:y_energy.shape[0], :y_energy.shape[0]] = K_ee
        K[:y_energy.shape[0], y_energy.shape[0]:] = K_ef
        K[y_energy.shape[0]:, :y_energy.shape[0]] = K_ef.T
        K[y_energy.shape[0]:, y_energy.shape[0]:] = K_ff

        try:  # Use Cholesky decomposition to build the lower triangular matrix
            self.L_ = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'noise' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise

        self.y_energy_and_force = np.vstack(
            (self.y_train_energy_, self.y_train_))

        # Calculate the alpha weights using the Cholesky method
        self.alpha_ = cho_solve((self.L_, True), self.y_energy_and_force)
        self.energy_alpha_ = None  # Used to distinguish pure energy fitting
        self.K = K
        self.energy_K = None  # Used to distinguish pure energy fitting
        self.fitted = ['force', 'energy']
        self.n_train = len(self.X_train_) + len(self.X_glob_train_)

        return self

    # Untested, log_marginal_linkelihood not working as for now
    def fit_energy(self, X_glob, y, ncores=1):
        """Fit a Gaussian process regression model using local energies.

        Args:
            X_glob (list of lists of arrays): list of grouped training configurations
            y (np.ndarray): training total energies
            ncores (int): number of CPU workers to use, default is 1

        """
        self.kernel_ = self.kernel
        self.X_glob_train_ = X_glob
        self.y_train_energy_ = np.reshape(y, (y.shape[0], 1))

        if self.optimizer is not None:  # TODO Debug
            logger.warning("Optimizer not yet implemented for energy training")
            '''
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
            '''
        else:
            pass
        '''
        self.log_marginal_likelihood_value_ = \
        self.log_marginal_likelihood(self.kernel_.theta)
        '''

        # Precompute quantities required for predictions which are independent
        # of actual query points
        self.energy_K = self.kernel_.calc_gram_e(self.X_glob_train_, ncores)
        self.energy_K[np.diag_indices_from(self.energy_K)] += self.noise

        try:  # Use Cholesky decomposition to build the lower triangular matrix
            self.L_ = cholesky(self.energy_K, lower=True)
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'noise' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise

        # Calculate the alpha weights using the Cholesky method
        self.energy_alpha_ = cho_solve((self.L_, True), self.y_train_energy_)
        self.K = None
        self.alpha_ = None
        self.fitted[1] = 'energy'
        self.n_train = len(self.y_train_energy_)
        self.X_train_ = None

        return self

    def predict(self, X, return_std=False, ncores=1):
        """Predict forces using the Gaussian process regression model

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True)

        Args:
            X (np.ndarray): Target configuration where the GP is evaluated
            return_std (bool): If True, the standard-deviation of the
                predictive distribution of the target configurations is
                returned along with the mean.

        Returns:
            y_mean (np.ndarray): Mean of predictive distribution at target configurations.
            y_std (np.ndarray): Standard deviation of predictive distribution at target
                configurations. Only returned when return_std is True.

        """

        # Unfitted; predict based on GP prior
        if not hasattr(self, "X_glob_train_") and not hasattr(self, "X_train_"):
            kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            logger.warning("No training data, predicting based on prior")
            if return_std:
                y_var = kernel.calc_diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

        else:  # Predict based on GP posterior
            if self.fitted == ['force', None]:  # Predict using force data
                K_trans = self.kernel_.calc(X, self.X_train_, ncores)
                y_mean = K_trans.dot(self.alpha_[:, 0])

            elif self.fitted == [None, 'energy']:  # Predict using energy data
                K_force_energy = self.kernel_.calc_ef(
                    self.X_glob_train_, X, ncores).T
                y_mean = K_force_energy.dot(self.energy_alpha_[:, 0])

            else:  # Predict using both force and energy data
                K_trans = self.kernel_.calc(X, self.X_train_, ncores)
                K_force_energy = self.kernel_.calc_ef(
                    self.X_glob_train_, X, ncores).T
                K = np.hstack((K_force_energy, K_trans))
                y_mean = K.dot(self.alpha_[:, 0])

            if return_std:  # TODO CHECK FOR ENERGY, FORCE and FORCE +ENERGY FIT
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
                    logger.warning("Predicted variances smaller than 0. "
                                   "Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0
                return np.reshape(y_mean, (int(y_mean.shape[0] / 3), 3)), np.reshape(np.sqrt(y_var),
                                                                                     (int(y_var.shape[0] / 3), 3))
            else:
                return np.reshape(y_mean, (int(y_mean.shape[0] / 3), 3))

    def predict_energy(self, X, return_std=False, ncores=1, mapping=False, **kwargs):
        """Predict energies from forces only using the Gaussian process regression model

        This function evaluates the GP energies for a set of test configurations.

        Args:
            X (np.ndarray): Target configurations where the GP is evaluated
            return_std (bool): If True, the standard-deviation of the
                predictive distribution of the target configurations is
                returned along with the mean.

        Returns:
            y_mean (np.ndarray): Mean of predictive distribution at target configurations.
            y_std (np.ndarray): Standard deviation of predictive distribution at target
                configurations. Only returned when return_std is True.

        """

        # Unfitted; predict based on GP prior
        if not hasattr(self, "X_glob_train_") and not hasattr(self, "X_train_"):
            kernel = self.kernel
            e_mean = np.zeros(len(X))
            logger.warning("No training data, predicting based on prior")
            if return_std:
                y_var = kernel.calc_diag_e(X)
                return e_mean, np.sqrt(e_var)
            else:
                return e_mean

        else:  # Predict based on GP posterior

            if self.fitted == ['force', None]:  # Predict using force data
                K_trans = self.kernel_.calc_ef(
                    X, self.X_train_, ncores, mapping, **kwargs)
                # Line 4 (y_mean = f_star)
                e_mean = K_trans.dot(self.alpha_[:, 0])

            elif self.fitted == [None, 'energy']:  # Predict using energy data
                K_energy = self.kernel_.calc_ee(
                    X, self.X_glob_train_, ncores, mapping, **kwargs)
                e_mean = K_energy.dot(self.energy_alpha_[:, 0])

            else:  # Predict using both force and energy data
                K_energy = self.kernel_.calc_ee(
                    X, self.X_glob_train_, ncores, mapping, **kwargs)
                K_energy_force = self.kernel_.calc_ef(
                    X, self.X_train_, ncores, mapping, **kwargs)
                K = np.hstack((K_energy, K_energy_force))
                e_mean = K.dot(self.alpha_[:, 0])

            if return_std:  # TODO CHECK FOR ENERGY, FORCE and FORCE +ENERGY FIT
                # compute inverse K_inv of K based on its Cholesky
                # decomposition L and its inverse L_inv
                L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
                K_inv = L_inv.dot(L_inv.T)
                # Compute variance of predictive distribution
                if self.fitted == ['force', None]:  # Predict using force data
                    e_var = self.kernel_.calc_diag_e(X)
                    fit = np.einsum(
                        "ij,ij->i", np.dot(K_trans, K_inv), K_trans)
                    e_var -= fit

                elif self.fitted == [None, 'energy']:  # Predict using force data
                    e_var = self.kernel_.calc_diag_e(X)
                    fit = np.einsum(
                        "ij,ij->i", np.dot(K_energy, K_inv), K_energy)
                    e_var -= fit

                else:  # Predict using force data
                    e_var = self.kernel_.calc_diag_e(X)
                    fit = np.einsum("ij,ij->i", np.dot(K, K_inv), K)
                    e_var -= fit

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                e_var_negative = e_var < 0
                if np.any(e_var_negative):
                    logger.warning("Predicted variances smaller than 0. "
                                   "Setting those variances to 0.")
                    e_var[e_var_negative] = 0.0
                return e_mean, np.sqrt(e_var)

            else:
                return e_mean

    # TODO: debug for energy and energy-force fitting
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

    def _constrained_optimization(self, obj_func, initial_theta,
                                  bounds):  # TODO: debug for energy and energy-force fitting
        if self.optimizer == "fmin_l_bfgs_b":
            theta_opt, func_min, convergence_dict = \
                fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
            if convergence_dict["warnflag"] != 0:
                logger.warning("fmin_l_bfgs_b terminated abnormally with the "
                               " state: %s" % convergence_dict)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min

    # TODO: debug for energy and energy-force fitting
    def pseudo_log_likelihood(self):
        """Returns pseudo log-likelihood of the training data.

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

        # kernel = self.kernel_.clone_with_theta(theta)
        kernel = self.kernel
        # kernel.theta = theta

        K = kernel.calc_gram(self.X_train_)

        K[np.diag_indices_from(K)] += self.noise

        Km1 = np.linalg.inv(K)

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        alphas = np.dot(Km1, y_train)
        pred_means = y_train - alphas / np.diag(Km1)
        pred_variances = 1. / np.diag(Km1)

        # Compute pseudo log-likelihood (compare Equation 5.10-5.12 of GPML)
        log_probabilities = - (pred_means - y_train) ** 2 / (2 * pred_variances) - 0.5 * np.log(
            pred_variances) - 0.5 * np.log(2 * np.pi)

        pseudo_log_likelihood = np.sum(log_probabilities)

        return pseudo_log_likelihood

    def save(self, filename):
        """Dump the current GP model for later use

        Args:
            filename (str): name of the file where to save the GP

        """

        output = [self.kernel_.kernel_name,
                  self.noise,
                  self.optimizer,
                  self.n_restarts_optimizer,
                  self.fitted,
                  self.alpha_,
                  self.K,
                  self.energy_alpha_,
                  self.energy_K,
                  self.X_train_,
                  self.X_glob_train_,
                  self.L_,
                  self.n_train]

        np.save(filename, output)

    def load(self, filename):
        """Load a saved GP model

        Args:
            filename (str): name of the file where the GP is saved

        """
        self.kernel.kernel_name, \
            self.noise, \
            self.optimizer, \
            self.n_restarts_optimizer, \
            self.fitted, \
            self.alpha_, \
            self.K, \
            self.energy_alpha_, \
            self.energy_K, \
            self.X_train_, \
            self.X_glob_train_, \
            self.L_, \
            self.n_train = np.load(filename, allow_pickle=True)

        self.kernel_ = self.kernel

        print('Loaded GP from file')


class TwoBodySingleSpeciesGP(GaussianProcess):

    def __init__(self, theta, noise=1e-10, optimizer=None, n_restarts_optimizer=0):
        kernel = kernels.TwoBodySingleSpeciesKernel(theta=theta)

        super().__init__(
            kernel=kernel, noise=noise, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer)

    def build_grid(self, dists, element1):
        num = len(dists)
        confs = np.zeros((num, 1, 5))

        confs[:, 0, 0] = dists
        confs[:, 0, 3], confs[:, 0, 4] = element1, element1

        grid_2b = self.predict_energy(confs)

        return interpolation.Spline1D(dists, grid_2b)


class ThreeBodySingleSpeciesGP(GaussianProcess):

    def __init__(self, theta, noise=1e-10, optimizer=None, n_restarts_optimizer=0):
        kernel = kernels.ThreeBodySingleSpeciesKernel(theta=theta)

        super().__init__(
            kernel=kernel, noise=noise, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer)

    def build_grid(self, dists, element1):
        """Function that builds and predicts energies on a cube of values"""

        num = len(dists)

        inds, r_ij_x, r_ki_x, r_ki_y = self.generate_triplets(dists)

        confs = np.zeros((len(r_ij_x), 2, 5))

        confs[:, 0, 0] = r_ij_x  # Element on the x axis
        confs[:, 1, 0] = r_ki_x  # Reshape into confs shape: this is x2
        confs[:, 1, 1] = r_ki_y  # Reshape into confs shape: this is y2

        # Permutations of elements

        confs[:, :, 3] = element1  # Central element is always element 1
        confs[:, 0, 4] = element1  # Element on the x axis is always element 2
        # Element on the xy plane is always element 3
        confs[:, 1, 4] = element1

        grid_3b = np.zeros((num, num, num))
        grid_3b[inds] = self.predict_energy(confs).flatten()

        for ind_i in range(num):
            for ind_j in range(ind_i + 1):
                for ind_k in range(ind_j + 1):
                    grid_3b[ind_i, ind_k, ind_j] = grid_3b[ind_i, ind_j, ind_k]
                    grid_3b[ind_j, ind_i, ind_k] = grid_3b[ind_i, ind_j, ind_k]
                    grid_3b[ind_j, ind_k, ind_i] = grid_3b[ind_i, ind_j, ind_k]
                    grid_3b[ind_k, ind_i, ind_j] = grid_3b[ind_i, ind_j, ind_k]
                    grid_3b[ind_k, ind_j, ind_i] = grid_3b[ind_i, ind_j, ind_k]

        return interpolation.Spline3D(dists, dists, dists, grid_3b)

    @staticmethod
    def generate_triplets(dists):
        d_ij, d_jk, d_ki = np.meshgrid(
            dists, dists, dists, indexing='ij', sparse=False, copy=True)

        # Valid triangles according to triangle inequality
        inds = np.logical_and(
            d_ij <= d_jk + d_ki, np.logical_and(d_jk <= d_ki + d_ij, d_ki <= d_ij + d_jk))

        # Utilizing permutation invariance
        inds = np.logical_and(np.logical_and(d_ij >= d_jk, d_jk >= d_ki), inds)

        # Element on the x axis
        r_ij_x = d_ij[inds]

        # Element on the xy plane
        r_ki_x = (d_ij[inds] ** 2 - d_jk[inds] ** 2 +
                  d_ki[inds] ** 2) / (2 * d_ij[inds])

        # using abs to avoid numerical error near to 0
        r_ki_y = np.sqrt(np.abs(d_ki[inds] ** 2 - r_ki_x ** 2))

        return inds, r_ij_x, r_ki_x, r_ki_y
