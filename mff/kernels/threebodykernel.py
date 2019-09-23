# -*- coding: utf-8 -*-

import logging
import numpy as np
from abc import ABCMeta, abstractmethod

from mff.kernels.base import Kernel

import theano.tensor as T
from theano import function, scan
import ray

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
        k3_ef_loc (object): Local Energy-force kernel function
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
                for conf1 in X1[i]:
                    for conf2 in X2[j]:
                        K[i, j] += self.k3_ee(conf1, conf2, self.theta[0], self.theta[1], self.theta[2])

        return K


    def calc_ee_single(self, X1, X2):
        """
        Calculate the energy-energy kernel between a sets of configurations and a global environment.
        
        Args:
            X1 (list): list of N1 Mx5 arrays containing xyz coordinates and atomic species
            X2 (list): list of N2 Mx5 arrays containing xyz coordinates and atomic species
            
        Returns:
            K (matrix): N1 x N2 matrix of the scalar-valued kernels 
       
       """
        K = np.zeros(X2.shape[0])
        for conf1 in X1:
            for j in np.arange(X2.shape[0]):
                for conf2 in X2[j]:
                    K[j] += self.k3_ee(conf1, conf2, self.theta[0], self.theta[1], self.theta[2])

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
        K = np.zeros((X1.shape[0], 3))
        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                for conf1 in X1[i]:
                    K[i, :] += self.k3_ef(conf1, X2[j], self.theta[0], self.theta[1], self.theta[2])

        return K


    def calc_ef_reverse(self, X1, X2):
        """
        Calculate the energy-force kernel between two sets of configurations.
        
        Args:
            X1 (list): list of N1 Mx5 arrays containing xyz coordinates and atomic species
            X2 (list): list of N2 Mx5 arrays containing xyz coordinates and atomic species
            
        Returns:
            K (matrix): N1 x N2*3 matrix of the vector-valued kernels 
       
       """
        K = np.zeros(X2.shape[0] * 3)
        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K[3 * j:3 * j + 3] += self.k3_ef(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

        return K


    def calc_ef_loc(self, X1, X2):
        """
        Calculate the local energy-force kernel between two sets of configurations.
        Used only during mapping since it is faster than calc_ef and equivalent in that case.
        
        Args:
            X1 (list): list of N1 Mx5 arrays containing xyz coordinates and atomic species
            X2 (list): list of N2 Mx5 arrays containing xyz coordinates and atomic species
            
        Returns:
            K (matrix): N1 x N2*3 matrix of the vector-valued kernels 
       
        """
        K = np.zeros((X1.shape[0], X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                    K[i, 3 * j:3 * j + 3] += self.k3_ef(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])                
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
                confs = []
                for i in np.arange(len(X)):
                    for j in np.arange(i + 1):
                        thislist = np.asarray([X[i], X[j]])
                        confs.append(thislist)
                n = len(confs)
                logger.info('Using %i cores for the 3-body force-force gram matrix calculation' % (nnodes))

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

                ray.init()
                # Using the dummy function that has a single argument
                result = np.array(ray.get([self.dummy_calc_ff.remote(self, clist[i]) for i in range(nnodes)]))
                ray.shutdown()

                result = np.concatenate(result).reshape((n, 3, 3))


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
    @ray.remote
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
                confs = []
                
                # Create a list of couple of configurations, of length len(conf)*(len(conf)+1)/2
                for i in np.arange(len(X)):
                    for j in np.arange(i + 1):
                        thislist = np.asarray([X[i], X[j]])
                        confs.append(thislist)
                        
                n = len(confs)
                import sys
                sys.setrecursionlimit(10000)
                logger.info('Using %i cores for the 3-body energy-energy gram matrix calculation' % (nnodes))

                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(nnodes + 1)
                factor = (n + (nnodes - 1)) / nnodes
                splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [confs[splitind[i]:splitind[i + 1]] for i in
                         np.arange(nnodes)]  # Shape is nnodes * (ntrain*(ntrain+1)/2)/nnodes * 2 * single_conf

                ray.init()
                # Using the dummy function that has a single argument
                result = np.array(ray.get([self.dummy_calc_ee.remote(self, clist[i]) for i in range(nnodes)]))
                ray.shutdown()
                result = np.concatenate(result).flatten()

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
                    for k, conf1 in enumerate(X[i]):
                        diag[i, i] += self.k3_ee(conf1, conf1, self.theta[0], self.theta[1], self.theta[2])
                        for conf2 in X[i][:k]:
                            diag[i, i] += 2.0*self.k3_ee(conf1, conf2, self.theta[0], self.theta[1], self.theta[2]) # *2 here to speed up the loop
                    for j in np.arange(i):
                        for conf1 in X[i]:
                            for conf2 in X[j]:
                                off_diag[i, j] += self.k3_ee(conf1, conf2, self.theta[0], self.theta[1], self.theta[2])

            gram = diag + off_diag + off_diag.T
            return gram

    # Used to simplify multiprocessing call    
    @ray.remote
    def dummy_calc_ee(self, array):

        result = np.zeros(len(array))
        for i in np.arange(len(array)):
            for conf1 in array[i][0]:
                for conf2 in array[i][1]:
                    result[i] += self.k3_ee(conf1, conf2, self.theta[0], self.theta[1], self.theta[2])

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
                confs = []
                for i in np.arange(len(X)):
                    for j in np.arange(len(X)):
                        thislist = np.asarray([X[i], X[j]])
                        confs.append(thislist)

                n = len(confs)
                import sys
                sys.setrecursionlimit(10000)
                logger.info('Using %i cores for the 3-body energy-force gram matrix calculation' % (nnodes))

                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(nnodes + 1)   # edges of the bins
                factor = (n + (nnodes - 1)) / nnodes
                splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [confs[splitind[i]:splitind[i + 1]] for i in
                         np.arange(nnodes)]  # Shape is nnodes * (ntrain*(ntrain+1)/2)/nnodes

                ray.init()
                # Using the dummy function that has a single argument
                result = np.array(ray.get([self.dummy_calc_ef.remote(self, clist[i]) for i in range(nnodes)]))
                ray.shutdown()
                result = np.concatenate(result).reshape((n, 1, 3))

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
    @ray.remote
    def dummy_calc_ef(self, array):
        result = np.zeros((len(array), 3))
        for i in np.arange(len(array)):
            for conf1 in array[i][0]:
                result[i] = self.k3_ef(conf1, array[i][1], self.theta[0], self.theta[1], self.theta[2])
        return result

    def calc_gram_ef_mixed(self, X, X_glob, nnodes=1, eval_gradient=False):
        """
        Calculate the energy-force gram matrix for a set of configurations X.
        This returns a non-symmetric matrix which is equal to the transpose of 
        the force-energy gram matrix.
        
        Args:
            X (list): list of N1 M1x5 arrays containing xyz coordinates and atomic species
            X_glob (list): list of N2 M2x5 arrays containing xyz coordinates and atomic species
            nnodes (int): Number of CPU nodes to use for multiprocessing (default is 1)
            eval_gradient (bool): if True, evaluate the gradient of the gram matrix
            
        Returns:
            gram (matrix): N2 x N1*3 gram matrix of the vector-valued kernels 
       
       """
        gram = np.zeros((X_glob.shape[0], X.shape[0] * 3))

        if eval_gradient:
            raise NotImplementedError('ERROR: GRADIENT NOT IMPLEMENTED YET')
        else:
            if nnodes > 1:  # Multiprocessing
                confs = []
                for i in np.arange(len(X_glob)):
                    for j in np.arange(len(X)):
                        thislist = np.asarray([X[i], X[j]])
                        confs.append(thislist)
                n = len(confs)
                import sys
                sys.setrecursionlimit(10000)
                logger.info('Using %i cores for the 2-body energy-force gram matrix calculation' % (nnodes))

                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(nnodes + 1)
                factor = (n + (nnodes - 1)) / nnodes
                splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [confs[splitind[i]:splitind[i + 1]] for i in
                         np.arange(nnodes)]  # Shape is nnodes * (ntrain*(ntrain+1)/2)/nnodes

                ray.init()
                # Using the dummy function that has a single argument
                result = np.array(ray.get([self.dummy_calc_ef_mixed.remote(self, clist[i]) for i in range(nnodes)]))
                ray.shutdown()
                result = np.concatenate(result).reshape((n, 1, 3))


                for i in np.arange(X_glob.shape[0]):
                    for j in np.arange(X.shape[0]):
                        gram[i, 3 * j:3 * j + 3] = result[j + i * X_glob.shape[0]]

            else:
                for i in np.arange(X_glob.shape[0]):
                    for j in np.arange(X.shape[0]):
                        for k in X_glob[i]:
                            gram[i, 3 * j:3 * j + 3] += self.k3_ef(k, X[j], self.theta[0], self.theta[1], self.theta[2])

            self.gram_ef = gram

            return gram

    # Used to simplify multiprocessing
    @ray.remote
    def dummy_calc_ef_mixed(self, array):

        result = np.zeros((len(array), 3))
        for i in np.arange(len(array)):
            for conf1 in array[i][0]:
                result[i] += self.k3_ef(conf1, array[i][1], self.theta[0], self.theta[1], self.theta[2])

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

        # --------------------------------------------------
        # RELATIVE DISTANCES TO CENTRAL VECTOR AND BETWEEN NEIGHBOURS
        # --------------------------------------------------

        # first and second configuration
        r1j = T.sqrt(T.sum((rho1s[:, :] - r1[None, :]) ** 2, axis=1))
        r2m = T.sqrt(T.sum((rho2s[:, :] - r2[None, :]) ** 2, axis=1))
        rjk = T.sqrt(T.sum((rho1s[None, :, :] - rho1s[:, None, :]) ** 2, axis=2))
        rmn = T.sqrt(T.sum((rho2s[None, :, :] - rho2s[:, None, :]) ** 2, axis=2))


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
        ker = k1n  + k2n  + k3n 

        cut_j = T.exp(-theta / T.abs_(rc - r1j)) * (0.5 * (T.sgn(rc - r1j) + 1))
        cut_jk = cut_j[:, None] * cut_j[None, :] *(
                  T.exp(-theta / T.abs_(rc - rjk[:, :])) *
                  (0.5 * (T.sgn(rc - rjk) + 1))[:, :])    

        cut_m = T.exp(-theta / T.abs_(rc - r2m)) * (0.5 * (T.sgn(rc - r2m) + 1))
        cut_mn = cut_m[:, None] * cut_m[None, :] *(
                  T.exp(-theta / T.abs_(rc - rmn[:, :])) *
                  (0.5 * (T.sgn(rc - rmn) + 1))[:, :])

        # --------------------------------------------------
        # REMOVE DIAGONAL ELEMENTS AND ADD CUTOFF
        # --------------------------------------------------

        # remove diagonal elements AND lower triangular ones from first configuration
        mask_jk = T.triu(T.ones_like(rjk)) - T.identity_like(rjk)

        # remove diagonal elements from second configuration
        mask_mn = T.ones_like(rmn) - T.identity_like(rmn)

        # Combine masks
        mask_jkmn = mask_jk[:, :, None, None] * mask_mn[None, None, :, :]

        # Apply mask and then apply cutoff functions
        ker = ker * mask_jkmn
        ker = T.sum(ker * cut_jk[:, :, None, None] * cut_mn[None, None, :, :])


        # --------------------------------------------------
        # FINAL FUNCTIONS
        # --------------------------------------------------

        # global energy energy kernel
        k_ee_fun = function([r1, r2, rho1, rho2, sig, theta, rc], ker, on_unused_input='ignore')

        # global energy force kernel
        k_ef_cut = T.grad(ker, r2)
        k_ef_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ef_cut, on_unused_input='ignore')
        
        # local force force kernel
        k_ff_cut = T.grad(ker, r1)
        k_ff_cut_der, updates = scan(lambda j, k_ff_cut, r2: T.grad(k_ff_cut[j], r2),
                                     sequences=T.arange(k_ff_cut.shape[0]), non_sequences=[k_ff_cut, r2])
        k_ff_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ff_cut_der, on_unused_input='ignore')

        # WRAPPERS (we don't want to plug the position of the central element every time)

        def k3_ee(conf1, conf2, sig, theta, rc):
            """
            Three body kernel for global energy-energy correlation

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
            Three body kernel for global energy-force correlation

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
            Three body kernel for local force-force correlation

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
        delta_alphas_jmkn = delta_alphasjm[:, None, :, None] * delta_alphasjm[None, :, None, :]
        
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

        ker_loc = k1n * delta_perm1 + k2n * delta_perm3 + k3n * delta_perm5

        # Faster version of cutoff (less calculations)
        cut_j = T.exp(-theta / T.abs_(rc - r1j)) * (0.5 * (T.sgn(rc - r1j) + 1))
        cut_jk = (cut_j[:, None] * cut_j[None, :] *
                  T.exp(-theta / T.abs_(rc - rjk[:, :])) *
                  (0.5 * (T.sgn(rc - rjk) + 1))[:, :])
        
        cut_jkl = (cut_j[:, None, None] * cut_j[None, :, None] * cut_j[None, None, :] *
                  T.exp(-theta / T.abs_(rc - rjk[:, :, None])) *
                  (0.5 * (T.sgn(rc - rjk) + 1))[:, :, None] * 
                  T.exp(-theta / T.abs_(rc - rjk[:, None, :])) *
                  (0.5 * (T.sgn(rc - rjk) + 1))[:, None, :] * 
                  T.exp(-theta / T.abs_(rc - rjk[None, :, :])) *
                  (0.5 * (T.sgn(rc - rjk) + 1))[None, :, :] )     
        
        cut_m = T.exp(-theta / T.abs_(rc - r2m)) * (0.5 * (T.sgn(rc - r2m) + 1))
        cut_mn = (cut_m[:, None] * cut_m[None, :] *
                  T.exp(-theta / T.abs_(rc - rmn[:, :])) *
                  (0.5 * (T.sgn(rc - rmn) + 1))[:, :])

        cut_mno = (cut_m[:, None, None] * cut_m[None, :, None] * cut_m[None, None, :] *
                  T.exp(-theta / T.abs_(rc - rmn[:, :, None])) *
                  (0.5 * (T.sgn(rc - rmn) + 1))[:, :, None] * 
                  T.exp(-theta / T.abs_(rc - rmn[:, None, :])) *
                  (0.5 * (T.sgn(rc - rmn) + 1))[:, None, :] * 
                  T.exp(-theta / T.abs_(rc - rmn[None, :, :])) *
                  (0.5 * (T.sgn(rc - rmn) + 1))[None, :, :] )     
        
        # --------------------------------------------------
        # REMOVE DIAGONAL ELEMENTS
        # --------------------------------------------------

        
        # remove diagonal elements AND lower triangular ones from first configuration
        mask_jk = T.triu(T.ones_like(rjk)) - T.identity_like(rjk)

        # remove diagonal elements from second configuration
        mask_mn = T.ones_like(rmn) - T.identity_like(rmn)

        # Combine masks
        mask_jkmn = mask_jk[:, :, None, None] * mask_mn[None, None, :, :]

        # Apply mask and then apply cutoff functions
        ker_loc = ker_loc * mask_jkmn
        ker_loc = T.sum(ker_loc * cut_jk[:, :, None, None] * cut_mn[None, None, :, :])
        
        
        # Calculate the global energy kernel
        # Only remove diagonal elements since the kernel is NOT permutational invariant due to 
        # the possible combination of chemical species: this is why there are two mask_mn here:
        se_jkmn = se_jkmn* mask_jk[:,:,None,None] * mask_mn[None,None,:,:]
        se_jkmn = se_jkmn * delta_alphas_jmkn
        
        k4n = se_jkmn[:,:,None,:,:,None] * se_jkmn[:,None,:,:,None,:] * se_jkmn[None,:,:,None,:,:]
        k4n = T.sum(k4n * cut_jkl[:, :, :, None, None, None] * cut_mno[None, None, None, :, :, :])
        
        ker_glob = ker_loc + k4n

        # --------------------------------------------------
        # FINAL FUNCTIONS
        # --------------------------------------------------

        # energy energy kernel
        k_ee_fun = function([r1, r2, rho1, rho2, sig, theta, rc], ker_glob, on_unused_input='ignore')

        # energy force kernel
        k_ef_cut = T.grad(ker_glob, r2)
        k_ef_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ef_cut, on_unused_input='ignore')

        # energy force kernel
        k_ef_cut_loc = T.grad(ker_loc, r2)
        k_ef_fun_loc = function([r1, r2, rho1, rho2, sig, theta, rc], k_ef_cut, on_unused_input='ignore')
        
        # force force kernel
        k_ff_cut = T.grad(ker_loc, r1)
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
