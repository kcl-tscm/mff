# -*- coding: utf-8 -*-

import logging
import numpy as np
import theano.tensor as T

from abc import ABCMeta, abstractmethod
from mff.kernels.base import Kernel
from theano import function, scan

from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class BaseExpThreeBody(Kernel, metaclass=ABCMeta):
    """ Many body kernel class
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
        k2_ef_loc (object): Local Energy-force kernel function
        k2_ff (object): Force-force kernel function
        
    """

    @abstractmethod
    def __init__(self, kernel_name, theta, bounds):
        super().__init__(kernel_name)
        self.theta = theta
        self.bounds = bounds
        self.k2_ee, self.k2_ef, self.k2_ef_loc, self.k2_ff, = self.compile_theano()

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
                    self.k2_ff(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2], self.theta[3])

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
                K[i, 3 * j:3 * j + 3] = self.k2_ef(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2], self.theta[3])

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
                K[i, 3 * j:3 * j + 3] = self.k2_ef_loc(
                    X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2], self.theta[3])
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
                K[i, j] = self.k2_ee(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2], self.theta[3])

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
                from pathos.pools import ProcessPool  # Import multiprocessing package
                confs = []

                # Build a list of all input pairs which matrix needs to be computed
                for i in np.arange(len(X)):
                    for j in np.arange(i + 1):
                        thislist = np.asarray([X[i], X[j]])
                        confs.append(thislist)
                n = len(confs)
                import sys
                sys.setrecursionlimit(10000)
                logger.info(
                    'Using %i cores for the 2-body force-force gram matrix calculation' % (nnodes))


                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(nnodes + 1)
                factor = (n + (nnodes - 1)) / nnodes
                splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [confs[splitind[i]:splitind[i + 1]] for i in
                         np.arange(nnodes)]  # Shape is nnodes * (ntrain*(ntrain+1)/2)/nnodes


                pool = ProcessPool(nodes=nnodes)  # Use pool multiprocessing
                # Using the dummy function that has a single argument
                result = np.array(pool.map(self.dummy_calc_ff, clist))
                result = np.concatenate(result).reshape((n, 3, 3))
                pool.close()
                pool.join()
                pool.clear()

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
                        self.k2_ff(X[i], X[i], self.theta[0], self.theta[1], self.theta[2], self.theta[3])
                    for j in np.arange(i):
                        off_diag[3 * i:3 * i + 3, 3 * j:3 * j + 3] = \
                            self.k2_ff(X[i], X[j], self.theta[0], self.theta[1], self.theta[2], self.theta[3])

            gram = diag + off_diag + off_diag.T  # The gram matrix is symmetric
            return gram

    # Used to simplify multiprocessing call
    def dummy_calc_ff(self, array):

        result = np.zeros((len(array), 3, 3))
        for i in np.arange(len(array)):
            result[i] = self.k2_ff(
                array[i][0], array[i][1], self.theta[0], self.theta[1], self.theta[2], self.theta[3])

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
                logger.info(
                    'Using %i cores for the 2-body energy-energy gram matrix calculation' % (nnodes))
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
                        self.k2_ee(X[i], X[i], self.theta[0], self.theta[1], self.theta[2], self.theta[3])
                    for j in np.arange(i):
                        off_diag[i, j] = \
                            self.k2_ee(X[i], X[j], self.theta[0], self.theta[1], self.theta[2], self.theta[3])

            gram = diag + off_diag + off_diag.T  # Gram matrix is symmetric

            return gram

    # Used to simplify multiprocessing call
    def dummy_calc_ee(self, array):

        result = np.zeros(len(array))
        for i in np.arange(len(array)):
            result[i] = self.k2_ee(array[i][0], array[i][1], self.theta[0], self.theta[1], self.theta[2], self.theta[3])

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
                logger.info(
                    'Using %i cores for the 2-body energy-force gram matrix calculation' % (nnodes))
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
                            self.k2_ef(X[i], X[j], self.theta[0], self.theta[1], self.theta[2], self.theta[3])

            self.gram_ef = gram

            return gram

    # Used to simplify multiprocessing
    def dummy_calc_ef(self, array):

        result = np.zeros((len(array), 3))
        for i in np.arange(len(array)):
            result[i] = self.k2_ef(
                array[i][0], array[i][1], self.theta[0], self.theta[1], self.theta[2], self.theta[3])

        return result

    def calc_diag(self, X):  # Calculate the diagonal of a force-force gram matrix

        diag = np.zeros((X.shape[0] * 3))
        for i in np.arange(X.shape[0]):
            diag[i * 3:(i + 1) * 3] = np.diag(self.k2_ff(X[i], X[i], self.theta[0], self.theta[1], self.theta[2], self.theta[3]))

        return diag

    def calc_diag_e(self, X):  # Calculate the diagonal of an energy-energy gram matrix

        diag = np.zeros((X.shape[0]))
        for i in np.arange(X.shape[0]):
            diag[i] = self.k2_ee(X[i], X[i], self.theta[0], self.theta[1], self.theta[2], self.theta[3])

        return diag
    
    @staticmethod
    @abstractmethod
    def compile_theano():
        return None, None, None, None
    
class ExpThreeBodySingleSpeciesKernel(BaseExpThreeBody):
    """Many body single species kernel.

    Args:
        theta[0] (float): lengthscale of the kernel
        theta[1] (float): decay rate of the cutoff function
        theta[2] (float): cutoff radius
        theta[3] (float): many-bodiness exponential term
        
    """

    def __init__(self, theta=(1., 1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        super().__init__(kernel_name='ManyBodySingleSpecies', theta=theta, bounds=bounds)

    @staticmethod
    def compile_theano():
        """
        This function generates theano compiled kernels for global energy and force learning

        The position of the atoms relative to the central one, and their chemical species
        are defined by a matrix of dimension Mx5 here called r1 and r2.

        Returns:
            k2_ee (func): energy-energy kernel
            k2_ef (func): energy-force kernel
            k2_ff (func): force-force kernel
        """

        logger.info("Started compilation of theano Many body single species kernels")
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
        chi = T.dscalar('chi')
        
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
        # BUILD THE KERNEL BETWEEN TWO CONFS
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
        ker_3_ab = k1n  + k2n  + k3n 

        # --------------------------------------------------
        # BUILD THE KERNEL WITHIN CONF A
        # --------------------------------------------------

        # Squared exp of differences
        se_1j1j = T.exp(-(r1j[:, None] - r1j[None, :]) ** 2 / (2 * sig ** 2))
        se_jkjk = T.exp(-(rjk[:, :, None, None] - rjk[None, None, :, :]) ** 2 / (2 * sig ** 2))
        se_jk1j = T.exp(-(rjk[:, :, None] - r1j[None, None, :]) ** 2 / (2 * sig ** 2))
        se_1jjk = T.exp(-(r1j[:, None, None] - rjk[None, :, :]) ** 2 / (2 * sig ** 2))

        # Kernel not summed (cyclic permutations)
        k1a = (se_1j1j[:, None, :, None] * se_1j1j[None, :, None, :] * se_jkjk)
        k2a = (se_1jjk[:, None, :, :] * se_jk1j[:, :, None, :] * se_1j1j[None, :, :, None])
        k3a = (se_1j1j[:, None, None, :] * se_jk1j[:, :, :, None] * se_1jjk[None, :, :, :])
                
        # final shape is M1 M1 M2 M2
        ker_3_aa = k1a  + k2a  + k3a 

        # --------------------------------------------------
        # BUILD THE KERNEL WITHIN CONF B
        # --------------------------------------------------

        # Squared exp of differences
        se_2m2m = T.exp(-(r2m[:, None] - r2m[None, :]) ** 2 / (2 * sig ** 2))
        se_mnmn = T.exp(-(rmn[:, :, None, None] - rmn[None, None, :, :]) ** 2 / (2 * sig ** 2))
        se_mn2m = T.exp(-(rmn[:, :, None] - r2m[None, None, :]) ** 2 / (2 * sig ** 2))
        se_2mmn = T.exp(-(r2m[:, None, None] - rmn[None, :, :]) ** 2 / (2 * sig ** 2))

        # Kernel not summed (cyclic permutations)
        k1b = (se_2m2m[:, None, :, None] * se_2m2m[None, :, None, :] * se_mnmn)
        k2b = (se_2mmn[:, None, :, :] * se_mn2m[:, :, None, :] * se_2m2m[None, :, :, None])
        k3b = (se_2m2m[:, None, None, :] * se_mn2m[:, :, :, None] * se_2mmn[None, :, :, :])
                
        # final shape is M1 M1 M2 M2
        ker_3_bb = k1b  + k2b  + k3b 


        # --------------------------------------------------
        # REMOVE DIAGONAL ELEMENTS AND ADD CUTOFF
        # --------------------------------------------------

        # remove diagonal elements AND lower triangular ones from first configuration
        mask_jk = T.triu(T.ones_like(rjk))# - T.identity_like(rjk)

        # remove diagonal elements from second configuration
        mask_mn = T.ones_like(rmn)# - T.identity_like(rmn)

        # Combine masks
        mask_jkmn = mask_jk[:, :, None, None] * mask_mn[None, None, :, :]
        mask_jkjk = mask_jk[:, :, None, None] * mask_jk[None, None, :, :]
        mask_mnmn = mask_mn[:, :, None, None] * mask_mn[None, None, :, :]

        # Apply mask
        ker_3_ab = ker_3_ab * mask_jkmn
        ker_3_ab = T.sum(ker_3_ab)

        ker_3_aa = ker_3_aa * mask_jkjk
        ker_3_aa = T.sum(ker_3_aa)

        ker_3_bb = ker_3_bb * mask_mnmn
        ker_3_bb = T.sum(ker_3_bb)

        # Get the eexponential kernel
        ker_loc = T.exp(-(ker_3_aa + ker_3_bb - 2*ker_3_ab)/chi)

        # --------------------------------------------------
        # FINAL FUNCTIONS
        # --------------------------------------------------

        # energy energy kernel
        k_ee_fun = function([r1, r2, rho1, rho2, sig, theta, rc, chi], ker_loc,
                            allow_input_downcast=False, on_unused_input='ignore')

        # local energy force kernel - used only for the mapping because there it is equivalent but faster
        k_ef_loc_cut = T.grad(ker_loc, r2)
        k_ef_fun_loc = function([r1, r2, rho1, rho2, sig, theta, rc, chi], k_ef_loc_cut, on_unused_input='ignore')
        
        # local force force kernel
        k_ff_cut = T.grad(ker_loc, r1)
        k_ff_cut_der, updates = scan(lambda j, k_ff_cut, r2: T.grad(k_ff_cut[j], r2),
                                     sequences=T.arange(k_ff_cut.shape[0]), non_sequences=[k_ff_cut, r2])
        k_ff_fun = function([r1, r2, rho1, rho2, sig, theta, rc, chi], k_ff_cut_der, on_unused_input='ignore')

        # WRAPPERS (we don't want to plug the position of the central element every time)

        def k2_ee(conf1, conf2, sig, theta, rc, chi):
            """
            Two body kernel for global energy-energy correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (float): scalar valued energy-energy 2-body kernel
                
            """
            return k_ee_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc, chi) 
        
        def k2_ef(conf1, conf2, sig, theta, rc, chi):
            """
            Two body kernel for global energy-force correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (array): 3x1 energy-force 2-body kernel
                
            """
            return -k_ef_fun_loc(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc, chi)

        def k2_ef_loc(conf1, conf2, sig, theta, rc, chi):
            """
            Two body kernel for local energy-force correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (array): 3x1 energy-force 2-body kernel
                
            """
            return -k_ef_fun_loc(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc, chi)
        
        def k2_ff(conf1, conf2, sig, theta, rc, chi):
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
            return k_ff_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc, chi)
        
        
        return k2_ee, k2_ef, k2_ef_loc, k2_ff

class ManyBodyTwoSpeciesKernel(BaseExpThreeBody):
    """Many body two species kernel.

    Args:
        theta[0] (float): lengthscale of the kernel
        theta[1] (float): decay rate of the cutoff function
        theta[2] (float): cutoff radius
        
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        super().__init__(kernel_name='ManyBody', theta=theta, bounds=bounds)

    @staticmethod
    def compile_theano():
        """
        This function generates theano compiled kernels for global energy and force learning

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
        alpha_1 = rho1[:, 3]#.flatten()
        alpha_2 = rho2[:, 3]#.flatten()
        alpha_j = rho1[:, 4]#.flatten()
        alpha_m = rho2[:, 4]#.flatten()
        
        # numerical kronecker
        def delta_alpha2(a1j, a2m):
            d = T.exp(-(a1j - a2m) ** 2 / (2 * 1e-5 ** 2))
            return d

        # matrices determining whether couples of atoms have the same atomic number
        delta_alphas12 = delta_alpha2(alpha_1[:, None], alpha_2[None, :])
        delta_alphasjm = delta_alpha2(alpha_j[:, None], alpha_m[None, :])
        delta_alphas1m = delta_alpha2(alpha_1[:, None], alpha_m[None, :])
        delta_alphasj2 = delta_alpha2(alpha_j[:, None], alpha_2[None, :])
        
        # distances of atoms wrt to the central one and wrt each other in 1 and 2
        r1j = T.sqrt(T.sum((rho1s[:, :] - r1[None, :]) ** 2, axis=1))
        r2m = T.sqrt(T.sum((rho2s[:, :] - r2[None, :]) ** 2, axis=1))
        rjk = T.sqrt(T.sum((rho1s[None, :, :] - rho1s[:, None, :]) ** 2, axis=2))
        rmn = T.sqrt(T.sum((rho2s[None, :, :] - rho2s[:, None, :]) ** 2, axis=2))
        
        # Get the squared exponential kernels
        se_jm = T.exp(-(r1j[:, None] - r2m[None, :]) ** 2 / (2 * sig ** 2))
        se_jkmn = T.exp(-(rjk[:, :, None, None] - rmn[None, None, :, :]) ** 2 / (2 * sig ** 2))
        se_jk2m = T.exp(-(rjk[:, :, None] - r2m[None, None, :]) ** 2 / (2 * sig ** 2))
        se_1jmn = T.exp(-(r1j[:, None, None] - rmn[None, :, :]) ** 2 / (2 * sig ** 2))

        # Define cutoff functions 
        cut_jkmn = (0.5 * (1 + T.sgn(rc - rjk[:, :, None, None]))) * (0.5 * (1 + T.sgn(rc - rmn[None, None, :, :]))) * \
                 (T.exp(-theta / abs(rc - rjk[:, :, None, None])) * T.exp(-theta / abs(rc - rmn[None, None, :, :])))
    
        cut_jm = (0.5 * (1 + T.sgn(rc - r1j[:, None]))) * (0.5 * (1 + T.sgn(rc - r2m[None, :]))) * \
                 (T.exp(-theta / abs(rc - r1j[:, None])) * T.exp(-theta / abs(rc - r2m[None, :])))
        
        cut_jkm = (0.5 * (1 + T.sgn(rc - rjk[:, :, None]))) * (0.5 * (1 + T.sgn(rc - r2m[None, None, :]))) * \
                 (T.exp(-theta / abs(rc - rjk[:, :, None])) * T.exp(-theta / abs(rc - r2m[None, None, :])))

        cut_jmn = (0.5 * (1 + T.sgn(rc - r1j[:, None, None]))) * (0.5 * (1 + T.sgn(rc - rmn[None, :, :]))) * \
                 (T.exp(-theta / abs(rc - r1j[:, None, None])) * T.exp(-theta / abs(rc - rmn[None, :, :])))
        
        # Apply cutoffs and chemical species masks
        se_jm = se_jm*cut_jm* (delta_alphas12 * delta_alphasjm + delta_alphas1m * delta_alphasj2)
        se_jkmn = se_jkmn*cut_jkmn * (
            delta_alphasjm[:,None,:,None] * delta_alphasjm[None,:,None,:] + delta_alphasjm[:,None,None,:] * delta_alphasjm[None,:,:,None])
        se_jk2m = se_jk2m*cut_jkm*(
            delta_alphasj2[:,None,:] * delta_alphasjm[None,:,:] + delta_alphasj2[None,:,:] * delta_alphasjm[:,None,:])
        se_1jmn = se_1jmn*cut_jmn*(
            delta_alphas1m[:,:,None] * delta_alphasjm[:,None,:] + delta_alphas1m[:,None,:] * delta_alphasjm[:,:,None])

        # Add mask of zeros to avoid double counting bonds
        mask_jk = T.triu(T.ones_like(rjk)) - T.identity_like(rjk)
        mask_mn = T.triu(T.ones_like(rmn)) - T.identity_like(rmn)
        
        # Apply masks
        se_jkmn = se_jkmn*mask_jk[:,:,None, None]*mask_mn[None, None,:,:]
        se_jk2m = se_jk2m*mask_jk[:,:,None]
        se_1jmn = se_1jmn*mask_mn[None,:,:]
        
        # Calculate global kernel k and local kernel kloc
        k = T.sum(se_jkmn) + T.sum(se_jk2m) + T.sum(se_1jmn) + T.sum(se_jm)
        kloc = T.sum(se_jm)

        # --------------------------------------------------
        # FINAL FUNCTIONS
        # --------------------------------------------------

        # global energy energy kernel
        k_ee_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k,
                            allow_input_downcast=False, on_unused_input='warn')

        # energy force kernel - Used to predict energies from forces
        k_ef = T.grad(k, r2)
        k_ef_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ef,
                            allow_input_downcast=False, on_unused_input='warn')

        # local energy force kernel - used only in mapping
        k_ef_loc = T.grad(kloc, r2)
        k_ef_loc_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ef_loc,
                            allow_input_downcast=False, on_unused_input='warn')
        
        # force force kernel - it uses only local atom pairs to avoid useless computation
        k_ff = T.grad(kloc, r1)
        k_ff_der, updates = scan(lambda j, k_ff, r2: T.grad(k_ff[j], r2),
                                 sequences=T.arange(k_ff.shape[0]), non_sequences=[k_ff, r2])

        k_ff_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ff_der,
                            allow_input_downcast=False, on_unused_input='warn')

#         # --------------------------------------------------
#         # WRAPPERS (we don't want to plug the position of the central element every time)
#         # --------------------------------------------------

        def k2_ee(conf1, conf2, sig, theta, rc):
            """
            Two body kernel for global energy-energy correlation

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
            Two body kernel for global energy-force correlation

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

        def k2_ef_loc(conf1, conf2, sig, theta, rc):
            """
            Two body kernel for local energy-force correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                theta (float): cutoff decay rate hyperparameter theta[1]
                rc (float): cutoff distance hyperparameter theta[2]

            Returns:
                kernel (array): 3x1 energy-force 2-body kernel
                
            """
            return -k_ef_loc_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)
        
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

        return k2_ee, k2_ef, k2_ef_loc, k2_ff