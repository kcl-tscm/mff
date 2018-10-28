# -*- coding: utf-8 -*-
"""
Two Body Kernel
===============

Module that contains the expressions for the 2-body single-species and
multi-species kernel. The module uses the Theano package to create the 
energy-energy, force-energy and force-force kernels through automatic
differentiation of the energy-energy kernel.
The module is used to calculate the energy-energy, energy-force and 
force-force gram matrices for the Gaussian processes, and supports 
multi processing.
The module is called by the gp.py script.

Example:

    >>> from twobodykernel import TwoBodySingleSpeciesKernel
    >>> kernel = kernels.TwoBodySingleSpeciesKernel(theta=[sigma, theta, r_cut])
    >>> ee_gram_matrix = kernel.calc_gram_e(training_configurations, number_nodes)

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
import logging
import numpy as np
import theano.tensor as T

from abc import ABCMeta, abstractmethod
from mff.kernels.base import Kernel
from theano import function, scan

logger = logging.getLogger(__name__)


class BaseTwoBody(Kernel, metaclass=ABCMeta):
    """ Two body kernel class
    Handles the functions common to the single-species and
    multi-species two-body kernels.

    Args:
        kernel_name (str): To choose between single- and two-species kernel
        theta[0] (float) : lengthscale of the kernel
        theta[1] (float) : decay rate of the cutoff function
        theta[2] (float) : cutoff radius
        bounds (list) : bounds of the kernel function.

    Attributes:
        k2_ee (object): Energy-energy kernel function
        k2_ef (object): Energy-force kernel function
        k2_ff (object): Force-force kernel function
        
    """

    @abstractmethod
    def __init__(self, kernel_name, theta, bounds):
        super().__init__(kernel_name)
        self.theta = theta
        self.bounds = bounds

        self.k2_ee, self.k2_ef, self.k2_ff = self.compile_theano()

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
                    self.k2_ff(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

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
                K[i, 3 * j:3 * j + 3] = self.k2_ef(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

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
                K[i, j] = self.k2_ee(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

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
            if nnodes > 1:  # Used for multiprocessing
                from pathos.multiprocessing import ProcessingPool  # Import multiprocessing package
                confs = []

                # Build a list of all input pairs which matrix needs to be computed
                for i in np.arange(len(X)):
                    for j in np.arange(i + 1):
                        thislist = np.asarray([X[i], X[j]])
                        confs.append(thislist)
                n = len(confs)
                import sys
                sys.setrecursionlimit(10000)
                logger.info('Using %i cores for the 2-body force-force gram matrix calculation' % (nnodes))
                pool = ProcessingPool(nodes=nnodes)  # Use pool multiprocessing

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
                        self.k2_ff(X[i], X[i], self.theta[0], self.theta[1], self.theta[2])
                    for j in np.arange(i):
                        off_diag[3 * i:3 * i + 3, 3 * j:3 * j + 3] = \
                            self.k2_ff(X[i], X[j], self.theta[0], self.theta[1], self.theta[2])

            gram = diag + off_diag + off_diag.T  # The gram matrix is symmetric
            return gram

    # Used to simplify multiprocessing call
    def dummy_calc_ff(self, array):

        result = np.zeros((len(array), 3, 3))
        for i in np.arange(len(array)):
            result[i] = self.k2_ff(array[i][0], array[i][1], self.theta[0], self.theta[1], self.theta[2])

        return result

    def calc_gram_e(self, X, nnodes=1, eval_gradient=False):
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
            if nnodes > 1:  # Used for multiprocessing
                from pathos.multiprocessing import ProcessingPool  # Import multiprocessing package
                confs = []

                # Build a list of all input pairs which matrix needs to be computed       
                for i in np.arange(len(X)):
                    for j in np.arange(i + 1):
                        thislist = np.asarray([X[i], X[j]])
                        confs.append(thislist)
                n = len(confs)
                import sys
                sys.setrecursionlimit(10000)
                logger.info('Using %i cores for the 2-body energy-energy gram matrix calculation' % (nnodes))
                pool = ProcessingPool(nodes=nnodes)

                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(nnodes + 1)
                factor = (n + (nnodes - 1)) / nnodes
                splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [confs[splitind[i]:splitind[i + 1]] for i in
                         np.arange(nnodes)]  # Shape is nnodes * (ntrain*(ntrain+1)/2)/nnodes

                # Using the dummy function that has a single argument
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
                        self.k2_ee(X[i], X[i], self.theta[0], self.theta[1], self.theta[2])
                    for j in np.arange(i):
                        off_diag[i, j] = \
                            self.k2_ee(X[i], X[j], self.theta[0], self.theta[1], self.theta[2])

            gram = diag + off_diag + off_diag.T  # Gram matrix is symmetric

            return gram

    # Used to simplify multiprocessing call
    def dummy_calc_ee(self, array):

        result = np.zeros(len(array))
        for i in np.arange(len(array)):
            result[i] = self.k2_ee(array[i][0], array[i][1], self.theta[0], self.theta[1], self.theta[2])

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
            if nnodes > 1:  # Multiprocessing
                from pathos.multiprocessing import ProcessingPool  # Import multiprocessing package
                confs = []
                for i in np.arange(len(X)):
                    for j in np.arange(len(X)):
                        thislist = np.asarray([X[i], X[j]])
                        confs.append(thislist)
                n = len(confs)
                import sys
                sys.setrecursionlimit(10000)
                logger.info('Using %i cores for the 2-body energy-force gram matrix calculation' % (nnodes))
                pool = ProcessingPool(nodes=nnodes)

                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(nnodes + 1)
                factor = (n + (nnodes - 1)) / nnodes
                splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [confs[splitind[i]:splitind[i + 1]] for i in
                         np.arange(nnodes)]  # Shape is nnodes * (ntrain*(ntrain+1)/2)/nnodes

                # Using the dummy function that has a single argument
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
                            self.k2_ef(X[i], X[j], self.theta[0], self.theta[1], self.theta[2])

            self.gram_ef = gram

            return gram

    # Used to simplify multiprocessing
    def dummy_calc_ef(self, array):

        result = np.zeros((len(array), 3))
        for i in np.arange(len(array)):
            result[i] = self.k2_ef(array[i][0], array[i][1], self.theta[0], self.theta[1], self.theta[2])

        return result

    def calc_diag(self, X):  # Calculate the diagonal of a force-force gram matrix

        diag = np.zeros((X.shape[0] * 3))
        for i in np.arange(X.shape[0]):
            diag[i * 3:(i + 1) * 3] = np.diag(self.k2_ff(X[i], X[i], self.theta[0], self.theta[1], self.theta[2]))

        return diag

    def calc_diag_e(self, X):  # Calculate the diagonal of an energy-energy gram matrix

        diag = np.zeros((X.shape[0]))
        for i in np.arange(X.shape[0]):
            diag[i] = self.k2_ee(X[i], X[i], self.theta[0], self.theta[1], self.theta[2])

        return diag

    @staticmethod
    @abstractmethod
    def compile_theano():
        return None, None, None


class TwoBodySingleSpeciesKernel(BaseTwoBody):
    """Two body single species kernel.

    Args:
        theta[0] (float): lengthscale of the kernel
        theta[1] (float): decay rate of the cutoff function
        theta[2] (float): cutoff radius
        
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        super().__init__(kernel_name='TwoBodySingleSpecies', theta=theta, bounds=bounds)

    @staticmethod
    def compile_theano():
        """
        This function generates theano compiled kernels for energy and force learning

        The position of the atoms relative to the central one, and their chemical species
        are defined by a matrix of dimension Mx5 here called r1 and r2.

        Returns:
            k2_ee (func): energy-energy kernel
            k2_ef (func): energy-force kernel
            k2_ff (func): force-force kernel
        """

        logger.info("Started compilation of theano two body single species kernels")
        # --------------------------------------------------
        # INITIAL DEFINITIONS
        # --------------------------------------------------

        # positions of central atoms
        r1, r2 = T.dvectors('r1d', 'r2d')
        # positions of neighbours
        rho1, rho2 = T.dmatrices('rho1', 'rho2')
        # lengthscale hyperparameter
        sig = T.dscalar('sig')
        # cutoff hyperparameters
        theta = T.dscalar('theta')
        rc = T.dscalar('rc')

        # positions of neighbours without chemical species (3D space assumed)
        rho1s = rho1[:, 0:3]
        rho2s = rho2[:, 0:3]

        # --------------------------------------------------
        # RELATIVE DISTANCES TO CENTRAL VECTOR
        # --------------------------------------------------

        # first and second configuration
        r1j = T.sqrt(T.sum((rho1s[:, :] - r1[None, :]) ** 2, axis=1))
        r2m = T.sqrt(T.sum((rho2s[:, :] - r2[None, :]) ** 2, axis=1))

        # Cutoff function
        k_ij = T.exp(-(r1j[:, None] - r2m[None, :]) ** 2 / (2 * sig ** 2))

        cut_ij = (0.5 * (1 + T.sgn(rc - r1j[:, None]))) * (0.5 * (1 + T.sgn(rc - r2m[None, :]))) * \
                 (T.exp(-theta / (rc - r1j[:, None])) * T.exp(-theta / (rc - r2m[None, :])))

        k_ij = k_ij * cut_ij

        # kernel
        k = T.sum(k_ij)

        # --------------------------------------------------
        # FINAL FUNCTIONS
        # --------------------------------------------------

        # energy energy kernel
        k_ee_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k,
                            allow_input_downcast=False, on_unused_input='warn')

        # energy force kernel - Used to predict energies from forces
        k_ef = T.grad(k, r2)
        k_ef_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ef,
                            allow_input_downcast=False, on_unused_input='warn')

        # force force kernel
        k_ff = T.grad(k, r1)
        k_ff_der, updates = scan(lambda j, k_ff, r2: T.grad(k_ff[j], r2),
                                 sequences=T.arange(k_ff.shape[0]), non_sequences=[k_ff, r2])

        k_ff_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ff_der,
                            allow_input_downcast=False, on_unused_input='warn')

        # --------------------------------------------------
        # WRAPPERS (we don't want to plug the position of the central element every time)
        # --------------------------------------------------

        def k2_ee(conf1, conf2, sig, theta, rc):
            """
            Two body kernel for energy-energy correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (float): scalar valued energy-energy 2-body kernel
                
            """
            return k_ee_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        def k2_ef(conf1, conf2, sig, theta, rc):
            """
            Two body kernel for energy-force correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (array): 3x1 energy-force 2-body kernel
                
            """
            return -k_ef_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        def k2_ff(conf1, conf2, sig, theta, rc):
            """
            Two body kernel for force-force correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (matrix): 3x3 force-force 2-body kernel
                
            """
            return k_ff_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        logger.info("Ended compilation of theano two body single species kernels")

        return k2_ee, k2_ef, k2_ff


class TwoBodyTwoSpeciesKernel(BaseTwoBody):
    """Two body two species kernel.

    Args:
        theta[0] (float): lengthscale of the kernel
        theta[1] (float): decay rate of the cutoff function
        theta[2] (float): cutoff radius
        
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        super().__init__(kernel_name='TwoBody', theta=theta, bounds=bounds)

    @staticmethod
    def compile_theano():
        """
        This function generates theano compiled kernels for energy and force learning

        The position of the atoms relative to the central one, and their chemical species
        are defined by a matrix of dimension Mx5 here called r1 and r2.

        Returns:
            k2_ee (func): energy-energy kernel
            k2_ef (func): energy-force kernel
            k2_ff (func): force-force kernel
        """

        logger.info("Started compilation of theano two body kernels")
        # --------------------------------------------------
        # INITIAL DEFINITIONS
        # --------------------------------------------------

        # positions of central atoms
        r1, r2 = T.dvectors('r1d', 'r2d')
        # positions of neighbours
        rho1, rho2 = T.dmatrices('rho1', 'rho2')
        # lengthscale hyperparameter
        sig = T.dscalar('sig')
        # cutoff hyperparameters
        theta = T.dscalar('theta')
        rc = T.dscalar('rc')

        # positions of neighbours without chemical species (3D space assumed)
        rho1s = rho1[:, 0:3]
        rho2s = rho2[:, 0:3]
        alpha_1 = rho1[:, 3:4].flatten()
        alpha_2 = rho2[:, 3:4].flatten()
        alpha_j = rho1[:, 4:5].flatten()
        alpha_m = rho2[:, 4:5].flatten()

        # --------------------------------------------------
        # RELATIVE DISTANCES TO CENTRAL VECTOR
        # --------------------------------------------------

        # first and second configuration
        r1j = T.sqrt(T.sum((rho1s[:, :] - r1[None, :]) ** 2, axis=1))
        r2m = T.sqrt(T.sum((rho2s[:, :] - r2[None, :]) ** 2, axis=1))

        # --------------------------------------------------
        # CHEMICAL SPECIES MASK
        # --------------------------------------------------

        # numerical kronecker
        def delta_alpha2(a1j, a2m):
            d = T.exp(-(a1j - a2m) ** 2 / (2 * 1e-5 ** 2))
            return d

        delta_alphas12 = delta_alpha2(alpha_1[:, None], alpha_2[None, :])
        delta_alphasjm = delta_alpha2(alpha_j[:, None], alpha_m[None, :])
        delta_alphas1m = delta_alpha2(alpha_1[:, None], alpha_m[None, :])
        delta_alphasj2 = delta_alpha2(alpha_j[:, None], alpha_2[None, :])

        # Cutoff function
        k_ij = (T.exp(-(r1j[:, None] - r2m[None, :]) ** 2 / (2 * sig ** 2)) * (
                delta_alphas12 * delta_alphasjm + delta_alphas1m * delta_alphasj2))

        cut_ij = (0.5 * (1 + T.sgn(rc - r1j[:, None]))) * (0.5 * (1 + T.sgn(rc - r2m[None, :]))) * \
                 (T.exp(-theta / (rc - r1j[:, None])) * T.exp(-theta / (rc - r2m[None, :])))

        k_ij = k_ij * cut_ij

        # kernel
        k = T.sum(k_ij)

        # --------------------------------------------------
        # FINAL FUNCTIONS
        # --------------------------------------------------

        # energy energy kernel
        k_ee_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k,
                            allow_input_downcast=False, on_unused_input='ignore')

        # energy force kernel
        k_ef = T.grad(k, r2)
        k_ef_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ef,
                            allow_input_downcast=False, on_unused_input='ignore')

        # force force kernel
        k_ff = T.grad(k, r1)
        k_ff_der, updates = scan(lambda j, k_ff, r2: T.grad(k_ff[j], r2),
                                 sequences=T.arange(k_ff.shape[0]), non_sequences=[k_ff, r2])

        k_ff_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ff_der,
                            allow_input_downcast=False, on_unused_input='ignore')

        # --------------------------------------------------
        # WRAPPERS (we don't want to plug the position of the central element every time)
        # --------------------------------------------------

        def k2_ee(conf1, conf2, sig, theta, rc):
            """
            Two body kernel for energy-energy correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (float): scalar valued energy-energy 2-body kernel
                
            """
            return k_ee_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        def k2_ef(conf1, conf2, sig, theta, rc):
            """
            Two body kernel for energy-force correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (array): 3x1 energy-force 2-body kernel
                
            """
            return -k_ef_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        def k2_ff(conf1, conf2, sig, theta, rc):
            """
            Two body kernel for energy-energy correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (matrix): 3x3 force-force 2-body kernel
                
            """
            return k_ff_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        logger.info("Ended compilation of theano two body kernels")

        return k2_ee, k2_ef, k2_ff
