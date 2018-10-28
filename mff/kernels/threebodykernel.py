# -*- coding: utf-8 -*-
"""
Three Body Kernel
=================

Module that contains the expressions for the 3-body single-species and
multi-species kernel. The module uses the Theano package to create the 
energy-energy, force-energy and force-force kernels through automatic
differentiation of the energy-energy kernel.
The module is used to calculate the energy-energy, energy-force and 
force-force gram matrices for the Gaussian processes, and supports 
multi processing.
The module is called by the gp.py script.

Example:

    >>> from threebodykernel import ThreeBodySingleSpeciesKernel
    >>> kernel = kernels.ThreeBodySingleSpeciesKernel(theta=[sigma, theta, r_cut])
    >>> ee_gram_matrix = kernel.calc_gram_e(training_configurations, number_nodes)

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
import logging
import numpy as np
from abc import ABCMeta, abstractmethod

from mff.kernels.base import Kernel

import theano.tensor as T
from theano import function, scan

logger = logging.getLogger(__name__)


class BaseThreeBody(Kernel, metaclass=ABCMeta):
    """ Three body kernel class
    Handles the functions common to the single-species and
    multi-species three-body kernels.

    Args:
        kernel_name (str): To choose between single- and two-species kernel
        theta[0] (float) : lengthscale of the kernel
        theta[1] (float) : decay rate of the cutoff function
        theta[2] (float) : cutoff radius
        bounds (list) : bounds of the kernel function.

    Attributes:
        k3_ee (object): Energy-energy kernel function
        k3_ef (object): Energy-force kernel function
        k3_ff (object): Force-force kernel function
        
    """

    @abstractmethod
    def __init__(self, kernel_name, theta, bounds):
        super().__init__(kernel_name)
        self.theta = theta
        self.bounds = bounds

        self.k3_ee, self.k3_ef, self.k3_ff = self.compile_theano()

    def calc(self, X1, X2):
        """
        Calculate the force-force kernel between two sets of configurations.
        
        Args:
            X1 (list): list of N1 Mx5 arrays containing xyz coordinates and atomic species
            X2 (list): list of N2 Mx5 arrays containing xyz coordinates and atomic species
            
        Returns:
            K (matrix): N1*3 x N2*3 matrix of the matrix-valued kernels 
       
        """
        K = np.zeros((X1.shape[0] * 3, X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K[3 * i:3 * i + 3, 3 * j:3 * j + 3] = \
                    self.k3_ff(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

        return K

    def calc_ee(self, X1, X2):
        """
        Calculate the energy-energy kernel between two sets of configurations.
        
        Args:
            X1 (list): list of N1 Mx5 arrays containing xyz coordinates and atomic species
            X2 (list): list of N2 Mx5 arrays containing xyz coordinates and atomic species
            
        Returns:
            K (matrix): N1 x N2 matrix of the scalar-valued kernels 
       
        """
        K = np.zeros((X1.shape[0], X2.shape[0]))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K[i, j] = self.k3_ee(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

        return K

    def calc_ef(self, X1, X2):
        """
        Calculate the energy-force kernel between two sets of configurations.
        
        Args:
            X1 (list): list of N1 Mx5 arrays containing xyz coordinates and atomic species
            X2 (list): list of N2 Mx5 arrays containing xyz coordinates and atomic species
            
        Returns:
            K (matrix): N1 x N2*3 matrix of the vector-valued kernels 
       
        """
        K = np.zeros((X1.shape[0], X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K[i, 3 * j:3 * j + 3] = self.k3_ef(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])
        return K

    def calc_gram(self, X, nnodes=1, eval_gradient=False):
        """
        Calculate the force-force gram matrix for a set of configurations X.
        
        Args:
            X (list): list of N Mx5 arrays containing xyz coordinates and atomic species
            nnodes (int): Number of CPU nodes to use for multiprocessing (default is 1)
            eval_gradient (bool): if True, evaluate the gradient of the gram matrix
            
        Returns:
            gram (matrix): N*3 x N*3 gram matrix of the matrix-valued kernels 
       
        """
        if eval_gradient:
            raise NotImplementedError('ERROR: GRADIENT NOT IMPLEMENTED YET')
        else:
            if nnodes > 1:
                from pathos.multiprocessing import ProcessingPool  # Import multiprocessing package
                confs = []
                for i in np.arange(len(X)):
                    for j in np.arange(i + 1):
                        thislist = np.asarray([X[i], X[j]])
                        confs.append(thislist)
                n = len(confs)
                logger.info('Using %i cores for the 3-body force-force gram matrix calculation' % (nnodes))
                pool = ProcessingPool(nodes=nnodes)

                import sys
                sys.setrecursionlimit(10000)

                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(nnodes + 1)
                factor = (n + (nnodes - 1)) / nnodes
                splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [confs[splitind[i]:splitind[i + 1]] for i in
                         np.arange(nnodes)]  # Shape is nnodes * (ntrain*(ntrain+1)/2)/nnodes

                # Using the dummy function that has a single argument
                result = np.array(pool.map(self.dummy_calc_ff, clist))
                result = np.concatenate(result).reshape((n, 3, 3))
                pool.close()
                pool.join()

                off_diag = np.zeros((len(X) * 3, len(X) * 3))
                diag = np.zeros((len(X) * 3, len(X) * 3))
                for i in np.arange(len(X)):
                    diag[3 * i:3 * i + 3, 3 * i:3 * i + 3] = result[i + i * (i + 1) // 2]
                    for j in np.arange(i):
                        off_diag[3 * i:3 * i + 3, 3 * j:3 * j + 3] = result[j + i * (i + 1) // 2]

            else:
                diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))
                off_diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))
                for i in np.arange(X.shape[0]):
                    diag[3 * i:3 * i + 3, 3 * i:3 * i + 3] = \
                        self.k3_ff(X[i], X[i], self.theta[0], self.theta[1], self.theta[2])
                    for j in np.arange(i):
                        off_diag[3 * i:3 * i + 3, 3 * j:3 * j + 3] = \
                            self.k3_ff(X[i], X[j], self.theta[0], self.theta[1], self.theta[2])

            gram = diag + off_diag + off_diag.T
            return gram

    # Used to simplify multiprocessing call
    def dummy_calc_ff(self, array):
        result = np.zeros((len(array), 3, 3))
        for i in np.arange(len(array)):
            result[i] = self.k3_ff(array[i][0], array[i][1], self.theta[0], self.theta[1], self.theta[2])
        return result

    def calc_gram_e(self, X, nnodes=1, eval_gradient=False):  # Untested
        """
        Calculate the energy-energy gram matrix for a set of configurations X.
        
        Args:
            X (list): list of N Mx5 arrays containing xyz coordinates and atomic species
            nnodes (int): Number of CPU nodes to use for multiprocessing (default is 1)
            eval_gradient (bool): if True, evaluate the gradient of the gram matrix
            
        Returns:
            gram (matrix): N x N gram matrix of the scalar-valued kernels 
       
        """
        if eval_gradient:
            raise NotImplementedError('ERROR: GRADIENT NOT IMPLEMENTED YET')
        else:
            if nnodes > 1:
                from pathos.multiprocessing import ProcessingPool  # Import multiprocessing package
                confs = []
                for i in np.arange(len(X)):
                    for j in np.arange(i + 1):
                        thislist = np.asarray([X[i], X[j]])
                        confs.append(thislist)
                n = len(confs)
                import sys
                sys.setrecursionlimit(10000)
                logger.info('Using %i cores for the 3-body energy-energy gram matrix calculation' % (nnodes))
                pool = ProcessingPool(nodes=nnodes)

                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(nnodes + 1)
                factor = (n + (nnodes - 1)) / nnodes
                splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [confs[splitind[i]:splitind[i + 1]] for i in
                         np.arange(nnodes)]  # Shape is nnodes * (ntrain*(ntrain+1)/2)/nnodes

                # Using the dummy function which has a single argument
                result = np.array(pool.map(self.dummy_calc_ee, clist))
                result = np.concatenate(result).flatten()
                pool.close()
                pool.join()

                off_diag = np.zeros((len(X), len(X)))
                diag = np.zeros((len(X), len(X)))
                for i in np.arange(len(X)):
                    diag[i, i] = result[i + i * (i + 1) // 2]
                    for j in np.arange(i):
                        off_diag[i, j] = result[j + i * (i + 1) // 2]

            else:
                diag = np.zeros((X.shape[0], X.shape[0]))
                off_diag = np.zeros((X.shape[0], X.shape[0]))
                for i in np.arange(X.shape[0]):
                    diag[i, i] = \
                        self.k3_ee(X[i], X[i], self.theta[0], self.theta[1], self.theta[2])
                    for j in np.arange(i):
                        off_diag[i, j] = \
                            self.k3_ee(X[i], X[j], self.theta[0], self.theta[1], self.theta[2])

            gram = diag + off_diag + off_diag.T
            return gram

    # Used to simplify multiprocessing call
    def dummy_calc_ee(self, array):
        result = np.zeros(len(array))
        for i in np.arange(len(array)):
            result[i] = self.k3_ee(array[i][0], array[i][1], self.theta[0], self.theta[1], self.theta[2])
        return result

    def calc_gram_ef(self, X, nnodes=1, eval_gradient=False):
        """
        Calculate the energy-force gram matrix for a set of configurations X.
        This returns a non-symmetric matrix which is equal to the transpose of 
        the force-energy gram matrix.
        
        Args:
            X (list): list of N Mx5 arrays containing xyz coordinates and atomic species
            nnodes (int): Number of CPU nodes to use for multiprocessing (default is 1)
            eval_gradient (bool): if True, evaluate the gradient of the gram matrix
            
        Returns:
            gram (matrix): N x N*3 gram matrix of the vector-valued kernels 
       
        """
        gram = np.zeros((X.shape[0], X.shape[0] * 3))

        if eval_gradient:
            raise NotImplementedError('ERROR: GRADIENT NOT IMPLEMENTED YET')
        else:
            if nnodes > 1:
                from pathos.multiprocessing import ProcessingPool  # Import multiprocessing package
                confs = []
                for i in np.arange(len(X)):
                    for j in np.arange(len(X)):
                        thislist = np.asarray([X[i], X[j]])
                        confs.append(thislist)
                n = len(confs)
                import sys
                sys.setrecursionlimit(10000)
                logger.info('Using %i cores for the 3-body energy-force gram matrix calculation' % (nnodes))
                pool = ProcessingPool(nodes=nnodes)

                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(nnodes + 1)
                factor = (n + (nnodes - 1)) / nnodes
                splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [confs[splitind[i]:splitind[i + 1]] for i in
                         np.arange(nnodes)]  # Shape is nnodes * (ntrain*(ntrain+1)/2)/nnodes

                # Using the dummy function which has a single argument
                result = np.array(pool.map(self.dummy_calc_ef, clist))
                result = np.concatenate(result).reshape((n, 1, 3))
                pool.close()
                pool.join()

                for i in np.arange(X.shape[0]):
                    for j in np.arange(X.shape[0]):
                        gram[i, 3 * j:3 * j + 3] = result[j + i * X.shape[0]]

            else:
                for i in np.arange(X.shape[0]):
                    for j in np.arange(X.shape[0]):
                        gram[i, 3 * j:3 * j + 3] = \
                            self.k3_ef(X[i], X[j], self.theta[0], self.theta[1], self.theta[2])

            self.gram_ef = gram
            return gram

    # Used to simplify multiprocessing call
    def dummy_calc_ef(self, array):
        result = np.zeros((len(array), 3))
        for i in np.arange(len(array)):
            result[i] = self.k3_ef(array[i][0], array[i][1], self.theta[0], self.theta[1], self.theta[2])
        return result

    def calc_diag(self, X):

        diag = np.zeros((X.shape[0] * 3))

        for i in np.arange(X.shape[0]):
            diag[i * 3:(i + 1) * 3] = np.diag(self.k3_ff(X[i], X[i], self.theta[0], self.theta[1], self.theta[2]))

        return diag

    def calc_diag_e(self, X):

        diag = np.zeros((X.shape[0]))

        for i in np.arange(X.shape[0]):
            diag[i] = self.k3_ee(X[i], X[i], self.theta[0], self.theta[1], self.theta[2])

        return diag

    @staticmethod
    @abstractmethod
    def compile_theano():
        return None, None, None


class ThreeBodyTwoSpeciesKernel(BaseThreeBody):
    """Three body two species kernel.

    Args:
        theta[0] (float): lengthscale of the kernel
        theta[1] (float): decay rate of the cutoff function
        theta[2] (float): cutoff radius
        
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        super().__init__(kernel_name='ThreeBody', theta=theta, bounds=bounds)

    @staticmethod
    def compile_theano():
        """
        This function generates theano compiled kernels for energy and force learning
        ker_jkmn_withcutoff = ker_jkmn #* cutoff_ikmn

        The position of the atoms relative to the centrla one, and their chemical species
        are defined by a matrix of dimension Mx5

        Returns:
            k3_ee (func): energy-energy kernel
            k3_ef (func): energy-force kernel
            k3_ff (func): force-force kernel
        """

        logger.info("Started compilation of theano three body kernels")

        # --------------------------------------------------
        # INITIAL DEFINITIONS
        # --------------------------------------------------

        # positions of central atoms
        r1, r2 = T.dvectors('r1d', 'r2d')
        # positions of neighbours
        rho1, rho2 = T.dmatrices('rho1', 'rho2')
        # hyperparameter
        sig = T.dscalar('sig')
        # cutoff hyperparameters
        theta = T.dscalar('theta')
        rc = T.dscalar('rc')

        # positions of neighbours without chemical species

        rho1s = rho1[:, 0:3]
        rho2s = rho2[:, 0:3]

        alpha_1 = rho1[:, 3].flatten()
        alpha_2 = rho2[:, 3].flatten()

        alpha_j = rho1[:, 4].flatten()
        alpha_m = rho2[:, 4].flatten()

        alpha_k = rho1[:, 4].flatten()
        alpha_n = rho2[:, 4].flatten()

        # --------------------------------------------------
        # RELATIVE DISTANCES TO CENTRAL VECTOR AND BETWEEN NEIGHBOURS
        # --------------------------------------------------

        # first and second configuration
        r1j = T.sqrt(T.sum((rho1s[:, :] - r1[None, :]) ** 2, axis=1))
        r2m = T.sqrt(T.sum((rho2s[:, :] - r2[None, :]) ** 2, axis=1))
        rjk = T.sqrt(T.sum((rho1s[None, :, :] - rho1s[:, None, :]) ** 2, axis=2))
        rmn = T.sqrt(T.sum((rho2s[None, :, :] - rho2s[:, None, :]) ** 2, axis=2))

        # --------------------------------------------------
        # CHEMICAL SPECIES MASK
        # --------------------------------------------------

        # numerical kronecker
        def delta_alpha2(a1j, a2m):
            d = np.exp(-(a1j - a2m) ** 2 / (2 * 0.00001 ** 2))
            return d

        # permutation 1

        delta_alphas12 = delta_alpha2(alpha_1[0], alpha_2[0])
        delta_alphasjm = delta_alpha2(alpha_j[:, None], alpha_m[None, :])
        delta_alphaskn = delta_alpha2(alpha_k[:, None], alpha_n[None, :])
        delta_alphas_jmkn = delta_alphasjm[:, None, :, None] * delta_alphaskn[None, :, None, :]

        delta_perm1 = delta_alphas12 * delta_alphas_jmkn

        #         # permutation 2 - not used in the current state

        #         delta_alphas12 = delta_alpha2(alpha_1[0], alpha_2[0])
        #         delta_alphasjn = delta_alpha2(alpha_j[:, None], alpha_n[None, :])
        #         delta_alphaskm = delta_alpha2(alpha_k[:, None], alpha_m[None, :])

        #         delta_alphas_jnkm = delta_alphasjn[:, None, None, :] * delta_alphaskm[None, :, :, None]

        #         delta_perm2 = delta_alphas12 * delta_alphas_jnkm

        # permutation 3
        delta_alphas1m = delta_alpha2(alpha_1[0, None], alpha_m[None, :]).flatten()
        delta_alphasjn = delta_alpha2(alpha_j[:, None], alpha_n[None, :])
        delta_alphask2 = delta_alpha2(alpha_k[:, None], alpha_2[None, 0]).flatten()

        delta_perm3 = delta_alphas1m[None, None, :, None] * delta_alphasjn[:, None, None, :] * \
                      delta_alphask2[None, :, None, None]

        #         # permutation 4 - not used in the current state
        #         delta_alphas1m = delta_alpha2(alpha_1[0, None], alpha_m[None, :]).flatten()
        #         delta_alphasj2 = delta_alpha2(alpha_j[:, None], alpha_2[None, 0]).flatten()
        #         delta_alphaskn = delta_alpha2(alpha_k[:, None], alpha_n[None, :])

        #         delta_perm4 = delta_alphas1m[None, None, :, None] * delta_alphaskn[None, :, None, :] * \
        #                       delta_alphasj2[:, None, None, None]

        # permutation 5
        delta_alphas1n = delta_alpha2(alpha_1[0, None], alpha_n[None, :]).flatten()
        delta_alphasj2 = delta_alpha2(alpha_j[:, None], alpha_2[None, 0]).flatten()
        delta_alphaskm = delta_alpha2(alpha_k[:, None], alpha_m[None, :])

        delta_perm5 = delta_alphas1n[None, None, None, :] * delta_alphaskm[None, :, :, None] * \
                      delta_alphasj2[:, None, None, None]

        #         # permutation 6 - not used in the current state
        #         delta_alphas1n = delta_alpha2(alpha_1[0, None], alpha_n[None, :]).flatten()
        #         delta_alphasjm = delta_alpha2(alpha_j[:, None], alpha_m[None, :])
        #         delta_alphask2 = delta_alpha2(alpha_k[:, None], alpha_2[None, 0]).flatten()

        #         delta_perm6 = delta_alphas1n[None, None, None, :] * delta_alphasjm[:, None, :, None] * \
        #                       delta_alphask2[None, :, None, None]

        # --------------------------------------------------
        # BUILD THE KERNEL
        # --------------------------------------------------

        # Squared exp of differences
        se_1j2m = T.exp(-(r1j[:, None] - r2m[None, :]) ** 2 / (2 * sig ** 2))
        se_jkmn = T.exp(-(rjk[:, :, None, None] - rmn[None, None, :, :]) ** 2 / (2 * sig ** 2))
        se_jk2m = T.exp(-(rjk[:, :, None] - r2m[None, None, :]) ** 2 / (2 * sig ** 2))
        se_1jmn = T.exp(-(r1j[:, None, None] - rmn[None, :, :]) ** 2 / (2 * sig ** 2))

        # Kernel not summed (cyclic permutations)
        k1n = (se_1j2m[:, None, :, None] * se_1j2m[None, :, None, :] * se_jkmn)
        k2n = (se_1jmn[:, None, :, :] * se_jk2m[:, :, None, :] * se_1j2m[None, :, :, None])
        k3n = (se_1j2m[:, None, None, :] * se_jk2m[:, :, :, None] * se_1jmn[None, :, :, :])

        # final shape is M1 M1 M2 M2

        ker_jkmn = k1n * delta_perm1 + k2n * delta_perm3 + k3n * delta_perm5

        # Faster version of cutoff (less calculations)
        cut_j = T.exp(-theta / T.abs_(rc - r1j)) * (0.5 * (T.sgn(rc - r1j) + 1))
        cut_jk = (cut_j[:, None] * cut_j[None, :] *
                  T.exp(-theta / T.abs_(rc - rjk[:, :])) *
                  (0.5 * (T.sgn(rc - rjk) + 1))[:, :])

        cut_m = T.exp(-theta / T.abs_(rc - r2m)) * (0.5 * (T.sgn(rc - r2m) + 1))
        cut_mn = (cut_m[:, None] * cut_m[None, :] *
                  T.exp(-theta / T.abs_(rc - rmn[:, :])) *
                  (0.5 * (T.sgn(rc - rmn) + 1))[:, :])

        ker_jkmn_withcutoff = ker_jkmn * cut_jk[:, :, None, None] * cut_mn[None, None, :, :]

        # --------------------------------------------------
        # REMOVE DIAGONAL ELEMENTS
        # --------------------------------------------------

        # remove diagonal elements AND lower triangular ones from first configuration
        mask_jk = T.triu(T.ones_like(rjk)) - T.identity_like(rjk)

        # remove diagonal elements from second configuration
        mask_mn = T.ones_like(rmn) - T.identity_like(rmn)

        mask_jkmn = mask_jk[:, :, None, None] * mask_mn[None, None, :, :]

        k_cutoff = T.sum(ker_jkmn_withcutoff * mask_jkmn)

        # --------------------------------------------------
        # FINAL FUNCTIONS
        # --------------------------------------------------

        # energy energy kernel
        k_ee_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_cutoff, on_unused_input='ignore')

        # energy force kernel
        k_ef_cut = T.grad(k_cutoff, r2)
        k_ef_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ef_cut, on_unused_input='ignore')

        # force force kernel
        k_ff_cut = T.grad(k_cutoff, r1)
        k_ff_cut_der, updates = scan(lambda j, k_ff_cut, r2: T.grad(k_ff_cut[j], r2),
                                     sequences=T.arange(k_ff_cut.shape[0]), non_sequences=[k_ff_cut, r2])
        k_ff_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ff_cut_der, on_unused_input='ignore')

        # WRAPPERS (we don't want to plug the position of the central element every time)

        def k3_ee(conf1, conf2, sig, theta, rc):
            """
            Three body kernel for energy-energy correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (float): scalar valued energy-energy 3-body kernel
                
            """
            return k_ee_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        def k3_ef(conf1, conf2, sig, theta, rc):
            """
            Three body kernel for energy-force correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (array): 3x1 energy-force 3-body kernel
                
            """
            return -k_ef_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        def k3_ff(conf1, conf2, sig, theta, rc):
            """
            Three body kernel for force-force correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (matrix): 3x3 force-force 3-body kernel
                
            """
            return k_ff_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        logger.info("Ended compilation of theano three body kernels")

        return k3_ee, k3_ef, k3_ff


class ThreeBodySingleSpeciesKernel(BaseThreeBody):
    """Three body two species kernel.

    Args:
        theta[0] (float): lengthscale of the kernel
        theta[1] (float): decay rate of the cutoff function
        theta[2] (float): cutoff radius
        
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        super().__init__(kernel_name='ThreeBodySingleSpecies', theta=theta, bounds=bounds)

    @staticmethod
    def compile_theano():
        """
        This function generates theano compiled kernels for energy and force learning
        ker_jkmn_withcutoff = ker_jkmn #* cutoff_ikmn

        The position of the atoms relative to the centrla one, and their chemical species
        are defined by a matrix of dimension Mx5

        Returns:
            k3_ee (func): energy-energy kernel
            k3_ef (func): energy-force kernel
            k3_ff (func): force-force kernel
        """

        logger.info("Started compilation of theano three body kernels")

        # --------------------------------------------------
        # INITIAL DEFINITIONS
        # --------------------------------------------------

        # positions of central atoms
        r1, r2 = T.dvectors('r1d', 'r2d')
        # positions of neighbours
        rho1, rho2 = T.dmatrices('rho1', 'rho2')
        # hyperparameter
        sig = T.dscalar('sig')
        # cutoff hyperparameters
        theta = T.dscalar('theta')
        rc = T.dscalar('rc')

        # positions of neighbours without chemical species

        rho1s = rho1[:, 0:3]
        rho2s = rho2[:, 0:3]

        alpha_1 = rho1[:, 3].flatten()
        alpha_2 = rho2[:, 3].flatten()

        alpha_j = rho1[:, 4].flatten()
        alpha_m = rho2[:, 4].flatten()

        alpha_k = rho1[:, 4].flatten()
        alpha_n = rho2[:, 4].flatten()

        # --------------------------------------------------
        # RELATIVE DISTANCES TO CENTRAL VECTOR AND BETWEEN NEIGHBOURS
        # --------------------------------------------------

        # first and second configuration
        r1j = T.sqrt(T.sum((rho1s[:, :] - r1[None, :]) ** 2, axis=1))
        r2m = T.sqrt(T.sum((rho2s[:, :] - r2[None, :]) ** 2, axis=1))
        rjk = T.sqrt(T.sum((rho1s[None, :, :] - rho1s[:, None, :]) ** 2, axis=2))
        rmn = T.sqrt(T.sum((rho2s[None, :, :] - rho2s[:, None, :]) ** 2, axis=2))

        # --------------------------------------------------
        # CHEMICAL SPECIES MASK
        # --------------------------------------------------

        # numerical kronecker
        def delta_alpha2(a1j, a2m):
            d = np.exp(-(a1j - a2m) ** 2 / (2 * 0.00001 ** 2))
            return d

        # permutation 1
        delta_alphas12 = delta_alpha2(alpha_1[0], alpha_2[0])
        delta_alphasjm = delta_alpha2(alpha_j[:, None], alpha_m[None, :])
        delta_alphaskn = delta_alpha2(alpha_k[:, None], alpha_n[None, :])
        delta_alphas_jmkn = delta_alphasjm[:, None, :, None] * delta_alphaskn[None, :, None, :]

        delta_perm1 = delta_alphas12 * delta_alphas_jmkn

        # permutation 3
        delta_alphas1m = delta_alpha2(alpha_1[0, None], alpha_m[None, :]).flatten()
        delta_alphasjn = delta_alpha2(alpha_j[:, None], alpha_n[None, :])
        delta_alphask2 = delta_alpha2(alpha_k[:, None], alpha_2[None, 0]).flatten()

        delta_perm3 = delta_alphas1m[None, None, :, None] * delta_alphasjn[:, None, None, :] * \
                      delta_alphask2[None, :, None, None]

        # permutation 5
        delta_alphas1n = delta_alpha2(alpha_1[0, None], alpha_n[None, :]).flatten()
        delta_alphasj2 = delta_alpha2(alpha_j[:, None], alpha_2[None, 0]).flatten()
        delta_alphaskm = delta_alpha2(alpha_k[:, None], alpha_m[None, :])
        delta_alphas_j21n = delta_alphasj2[:, None] * delta_alphas1n[None, :]

        delta_perm5 = delta_alphas1n[None, None, None, :] * delta_alphaskm[None, :, :, None] * \
                      delta_alphasj2[:, None, None, None]

        # --------------------------------------------------
        # BUILD THE KERNEL
        # --------------------------------------------------

        # Squared exp of differences
        se_1j2m = T.exp(-(r1j[:, None] - r2m[None, :]) ** 2 / (2 * sig ** 2))
        se_jkmn = T.exp(-(rjk[:, :, None, None] - rmn[None, None, :, :]) ** 2 / (2 * sig ** 2))
        se_jk2m = T.exp(-(rjk[:, :, None] - r2m[None, None, :]) ** 2 / (2 * sig ** 2))
        se_1jmn = T.exp(-(r1j[:, None, None] - rmn[None, :, :]) ** 2 / (2 * sig ** 2))

        # Kernel not summed (cyclic permutations)
        k1n = (se_1j2m[:, None, :, None] * se_1j2m[None, :, None, :] * se_jkmn)
        k2n = (se_1jmn[:, None, :, :] * se_jk2m[:, :, None, :] * se_1j2m[None, :, :, None])
        k3n = (se_1j2m[:, None, None, :] * se_jk2m[:, :, :, None] * se_1jmn[None, :, :, :])

        # final shape is M1 M1 M2 M2
        ker_jkmn = k1n * delta_perm1 + k2n * delta_perm3 + k3n * delta_perm5

        cut_j = T.exp(-theta / T.abs_(rc - r1j)) * (0.5 * (T.sgn(rc - r1j) + 1))
        cut_jk = (cut_j[:, None] * cut_j[None, :] *
                  T.exp(-theta / T.abs_(rc - rjk[:, :])) *
                  (0.5 * (T.sgn(rc - rjk) + 1))[:, :])

        cut_m = T.exp(-theta / T.abs_(rc - r2m)) * (0.5 * (T.sgn(rc - r2m) + 1))
        cut_mn = (cut_m[:, None] * cut_m[None, :] *
                  T.exp(-theta / T.abs_(rc - rmn[:, :])) *
                  (0.5 * (T.sgn(rc - rmn) + 1))[:, :])

        ker_jkmn_withcutoff = ker_jkmn * cut_jk[:, :, None, None] * cut_mn[None, None, :, :]

        # --------------------------------------------------
        # REMOVE DIAGONAL ELEMENTS
        # --------------------------------------------------

        # remove diagonal elements AND lower triangular ones from first configuration
        mask_jk = T.triu(T.ones_like(rjk)) - T.identity_like(rjk)

        # remove diagonal elements from second configuration
        mask_mn = T.ones_like(rmn) - T.identity_like(rmn)

        mask_jkmn = mask_jk[:, :, None, None] * mask_mn[None, None, :, :]

        k_cutoff = T.sum(ker_jkmn_withcutoff * mask_jkmn)

        # --------------------------------------------------
        # FINAL FUNCTIONS
        # --------------------------------------------------

        # energy energy kernel
        k_ee_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_cutoff, on_unused_input='ignore')

        # energy force kernel
        k_ef_cut = T.grad(k_cutoff, r2)
        k_ef_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ef_cut, on_unused_input='ignore')

        # force force kernel
        k_ff_cut = T.grad(k_cutoff, r1)
        k_ff_cut_der, updates = scan(lambda j, k_ff_cut, r2: T.grad(k_ff_cut[j], r2),
                                     sequences=T.arange(k_ff_cut.shape[0]), non_sequences=[k_ff_cut, r2])
        k_ff_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ff_cut_der, on_unused_input='ignore')

        # WRAPPERS (we don't want to plug the position of the central element every time)

        def k3_ee(conf1, conf2, sig, theta, rc):
            """
            Three body kernel for energy-energy correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (float): scalar valued energy-energy 3-body kernel
                
            """
            return k_ee_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        def k3_ef(conf1, conf2, sig, theta, rc):
            """
            Three body kernel for energy-force correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (array): 3x1 energy-force 3-body kernel
                
            """
            return -k_ef_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        def k3_ff(conf1, conf2, sig, theta, rc):
            """
            Three body kernel for force-force correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (matrix): 3x3 force-force 3-body kernel
                
            """
            return k_ff_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        logger.info("Ended compilation of theano three body kernels")

        return k3_ee, k3_ef, k3_ff
