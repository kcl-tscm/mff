# -*- coding: utf-8 -*-

import logging
import os.path
import pickle
from abc import ABCMeta, abstractmethod

import numpy as np

from mff.kernels.base import Kernel, Mffpath

logger = logging.getLogger(__name__)


def dummy_calc_ff(data):
    """ Function used when multiprocessing.
    Args:
        data (list of objects): contains all the information required
            for the computation of the kernel values
    
    Returns:
        result (array): the computed kernel values

    """

    array, theta0, theta1, theta2, kertype = data
    if kertype == "single":
        with open(Mffpath / "k3_ff_s.pickle", 'rb') as f:
            fun = pickle.load(f)
    elif kertype == "multi":
        with open(Mffpath / "k3_ff_m.pickle", 'rb') as f:
            fun = pickle.load(f)
    result = np.zeros((len(array), 3, 3))
    for i in range(len(array)):
        result[i] = fun(np.zeros(3), np.zeros(3), array[i][0],
                        array[i][1],  theta0, theta1, theta2)
    return result


def dummy_calc_ee(data):
    """ Function used when multiprocessing.
    Args:
        data (list of objects): contains all the information required
            for the computation of the kernel values
    
    Returns:
        result (array): the computed kernel values

    """

    array, theta0, theta1, theta2, kertype, mapping = data
    if kertype == "single":
        with open(Mffpath / "k3_ee_s.pickle", 'rb') as f:
            fun = pickle.load(f)
    elif kertype == "multi":
        with open(Mffpath / "k3_ee_m.pickle", 'rb') as f:
            fun = pickle.load(f)
    result = np.zeros(len(array))

    if not mapping:
        for i in range(len(array)):
            for conf1 in array[i][0]:
                for conf2 in array[i][1]:
                    result[i] += 1/9.0*fun(np.zeros(3), np.zeros(3),
                                     conf1, conf2, theta0, theta1, theta2)
    else:
        for i in range(len(array)):
            for conf2 in array[i][1]:
                result[i] += 1/3.0*fun(np.zeros(3), np.zeros(3),
                                 array[i][0], conf2, theta0, theta1, theta2)

    return result


def dummy_calc_ef(data):
    """ Function used when multiprocessing.
    Args:
        data (list of objects): contains all the information required
            for the computation of the kernel values
    
    Returns:
        result (array): the computed kernel values

    """
    
    array, theta0, theta1, theta2, kertype, mapping = data
    if kertype == "single":
        with open(Mffpath / "k3_ef_s.pickle", 'rb') as f:
            fun = pickle.load(f)
    elif kertype == "multi":
        with open(Mffpath / "k3_ef_m.pickle", 'rb') as f:
            fun = pickle.load(f)
    result = np.zeros((len(array), 3))
    if not mapping:
        for i in range(len(array)):
            conf2 = np.array(array[i][1], dtype='float')
            for conf1 in array[i][0]:
                conf1 = np.array(conf1, dtype='float')
                result[i] += -1/3.0*fun(np.zeros(3), np.zeros(3),
                                  conf1, conf2,  theta0, theta1, theta2)
    else:
        for i in range(len(array)):
            conf2 = np.array(array[i][1], dtype='float')
            conf1 = np.array(array[i][0], dtype='float')
            result[i] += -fun(np.zeros(3), np.zeros(3), conf1,
                              conf2,  theta0, theta1, theta2)
    return result


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

    def calc(self, X1, X2, ncores=1):
        """
        Calculate the energy-force kernel between two sets of configurations.

        Args:
            X1 (list): list of N1 Mx5 arrays containing xyz coordinates and atomic species
            X2 (list): list of N2 Mx5 arrays containing xyz coordinates and atomic species

        Returns:
            K (matrix): N2*3 matrix of the vector-valued kernels 

        """
        ker = np.zeros((len(X1) * 3, len(X2) * 3))

        if ncores > 1:

            confs = [[x1, x2] for x1 in X1 for x2 in X2]
            n = len(confs)
            import sys
            sys.setrecursionlimit(100000)
            logger.info(
                'Using %i cores for the 3-body force-force kernel calculation' % (ncores))

            # Way to split the kernels functions to compute evenly across the nodes
            splitind = np.zeros(ncores + 1)
            factor = (n + (ncores - 1)) / ncores
            splitind[1:-1] = [(i + 1) * factor for i in range(ncores - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [[confs[splitind[i]:splitind[i + 1]], self.theta[0], self.theta[1], self.theta[2],
                      self.type] for i in range(ncores)]  # Shape is ncores * (ntrain*(ntrain+1)/2)/ncores
            del confs
            import multiprocessing as mp
            pool = mp.Pool(ncores)
            result = pool.map(dummy_calc_ff, clist)
            del clist
            pool.close()
            pool.join()

            result = np.concatenate(result).reshape((n, 3, 3))
            for i in range(len(X1)):
                for j in range(len(X2)):
                    ker[i * 3: i * 3 + 3, 3 * j:3 * j +
                        3] = result[(j + i * len(X2))]
            del result

        else:
            for i, conf1 in enumerate(X1):
                for j, conf2 in enumerate(X2):
                    ker[i * 3:i * 3 + 3, 3 * j:3 * j + 3] += self.k3_ff(
                        conf1, conf2, self.theta[0], self.theta[1], self.theta[2])

        return ker

    def calc_ef(self, X_glob, X, ncores=1, mapping=False):
        """
        Calculate the energy-force kernel between two sets of configurations.

        Args:
            X1 (list): list of N1 Mx5 arrays containing xyz coordinates and atomic species
            X2 (list): list of N2 Mx5 arrays containing xyz coordinates and atomic species

        Returns:
            K (matrix): N2*3 matrix of the vector-valued kernels 

        """
        ker = np.zeros((len(X_glob), len(X) * 3))

        if ncores > 1:
            confs = [[x1, x2] for x1 in X_glob for x2 in X]
            n = len(confs)
            import sys
            sys.setrecursionlimit(100000)
            logger.info(
                'Using %i cores for the 3-body energy-force kernel calculation' % (ncores))

            # Way to split the kernels functions to compute evenly across the nodes
            splitind = np.zeros(ncores + 1)
            factor = (n + (ncores - 1)) / ncores
            splitind[1:-1] = [(i + 1) * factor for i in range(ncores - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [[confs[splitind[i]:splitind[i + 1]], self.theta[0], self.theta[1], self.theta[2],
                      self.type, mapping] for i in range(ncores)]  # Shape is ncores * (ntrain*(ntrain+1)/2)/ncores
            del confs
            import multiprocessing as mp
            pool = mp.Pool(ncores)
            result = pool.map(dummy_calc_ef, clist)
            del clist
            pool.close()
            pool.join()
            result = np.vstack(np.asarray(result))

            for i in range(len(X_glob)):
                for j in range(len(X)):
                    ker[i, 3 * j:3 * j + 3] = result[(j + i * len(X))]

        else:
            if not mapping:
                for i, x1 in enumerate(X_glob):
                    for j, conf2 in enumerate(X):
                        for conf1 in x1:
                            ker[i, 3 * j:3 * j + 3] += 1/3.0*self.k3_ef(
                                conf1, conf2, self.theta[0], self.theta[1], self.theta[2])
            else:
                for i, conf1 in enumerate(X_glob):
                    for j, conf2 in enumerate(X):
                        ker[i, 3 * j:3 * j + 3] += self.k3_ef(
                            conf1, conf2, self.theta[0], self.theta[1], self.theta[2])

        return ker

    def calc_ee(self, X1, X2, ncores=1, mapping=False):
        """
        Calculate the energy-energy kernel between two global environments.

        Args:
            X1 (list): list of N1 Mx5 arrays containing xyz coordinates and atomic species
            X2 (list): list of N2 Mx5 arrays containing xyz coordinates and atomic species

        Returns:
            K (matrix): N1 x N2 matrix of the scalar-valued kernels 

       """
        if ncores > 1:  # Used for multiprocessing

            # Build a list of all input pairs which matrix needs to be computed
            confs = [[x1, x2] for x1 in X1 for x2 in X2]

            n = len(confs)
            import sys
            sys.setrecursionlimit(100000)
            logger.info(
                'Using %i cores for the 3-body energy-energy kernel calculation' % (ncores))

            # Way to split the kernels functions to compute evenly across the nodes
            splitind = np.zeros(ncores + 1)
            factor = (n + (ncores - 1)) / ncores
            splitind[1:-1] = [(i + 1) * factor for i in range(ncores - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [[confs[splitind[i]:splitind[i + 1]], self.theta[0], self.theta[1], self.theta[2],
                      self.type, mapping] for i in range(ncores)]  # Shape is ncores * (ntrain*(ntrain+1)/2)/ncores

            del confs

            import multiprocessing as mp
            pool = mp.Pool(ncores)
            result = pool.map(dummy_calc_ee, clist)

            del clist
            pool.close()
            pool.join()
            result = np.concatenate(result).ravel()

            ker = np.zeros((len(X1), len(X2)))
            for i in range(len(X1)):
                for j in range(len(X2)):
                    ker[i, j] = result[j + i*len(X2)]

            del result 
            
        else:
            if not mapping:
                ker = np.zeros((len(X1), len(X2)))
                for i, x1 in enumerate(X1):
                    for j, x2 in enumerate(X2):
                        for conf1 in x1:
                            for conf2 in x2:
                                ker[i, j] += 1/9.0*self.k3_ee(conf1, conf2,
                                                        self.theta[0], self.theta[1], self.theta[2])
            else:
                ker = np.zeros((len(X1), len(X2)))
                for i, conf1 in enumerate(X1):
                    for j, x2 in enumerate(X2):
                        for conf2 in x2:
                            ker[i, j] += 1/3.0*self.k3_ee(conf1, conf2,
                                                    self.theta[0], self.theta[1], self.theta[2])

        return ker

    def calc_gram(self, X, ncores=1, eval_gradient=False):
        """
        Calculate the force-force gram matrix for a set of configurations X.

        Args:
            X (list): list of N Mx5 arrays containing xyz coordinates and atomic species
            ncores (int): Number of CPU nodes to use for multiprocessing (default is 1)
            eval_gradient (bool): if True, evaluate the gradient of the gram matrix

        Returns:
            gram (matrix): N*3 x N*3 gram matrix of the matrix-valued kernels 

        """
        if eval_gradient:
            raise NotImplementedError('ERROR: GRADIENT NOT IMPLEMENTED YET')
        else:
            if ncores > 1:
                confs = [[X[i], X[j]] for i in range(len(X)) for j in range(i + 1)]

                n = len(confs)
                logger.info(
                    'Using %i cores for the 3-body force-force gram matrix calculation' % (ncores))

                import sys
                sys.setrecursionlimit(100000)

                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(ncores + 1)
                factor = (n + (ncores - 1)) / ncores
                splitind[1:-1] = [(i + 1) *
                                  factor for i in range(ncores - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [[confs[splitind[i]:splitind[i + 1]], self.theta[0], self.theta[1], self.theta[2],
                          self.type] for i in range(ncores)]  # Shape is ncores * (ntrain*(ntrain+1)/2)/ncores
                del confs
                import multiprocessing as mp
                pool = mp.Pool(ncores)
                result = pool.map(dummy_calc_ff, clist)
                del clist
                pool.close()
                pool.join()

                result = np.concatenate(result).reshape((n, 3, 3))
                off_diag = np.zeros((len(X) * 3, len(X) * 3))
                diag = np.zeros((len(X) * 3, len(X) * 3))
                for i in range(len(X)):
                    diag[3 * i:3 * i + 3, 3 * i:3 * i +
                         3] = result[i + i * (i + 1) // 2]
                    for j in range(i):
                        off_diag[3 * i:3 * i + 3, 3 * j:3 *
                                 j + 3] = result[j + i * (i + 1) // 2]
                del result

            else:
                diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))
                off_diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))
                for i in range(X.shape[0]):
                    diag[3 * i:3 * i + 3, 3 * i:3 * i + 3] = \
                        self.k3_ff(X[i], X[i], self.theta[0],
                                   self.theta[1], self.theta[2])
                    for j in range(i):
                        off_diag[3 * i:3 * i + 3, 3 * j:3 * j + 3] = \
                            self.k3_ff(X[i], X[j], self.theta[0],
                                       self.theta[1], self.theta[2])

            gram = diag + off_diag + off_diag.T
            del diag, off_diag
            return gram

    def calc_gram_e(self, X, ncores=1, eval_gradient=False):  # Untested
        """
        Calculate the energy-energy gram matrix for a set of configurations X.

        Args:
            X (list): list of N Mx5 arrays containing xyz coordinates and atomic species
            ncores (int): Number of CPU nodes to use for multiprocessing (default is 1)
            eval_gradient (bool): if True, evaluate the gradient of the gram matrix

        Returns:
            gram (matrix): N x N gram matrix of the scalar-valued kernels 

        """
        if eval_gradient:
            raise NotImplementedError('ERROR: GRADIENT NOT IMPLEMENTED YET')
        else:
            if ncores > 1:
                # Build a list of all input pairs which matrix needs to be computed
                confs = [[X[i], X[j]] for i in range(len(X)) for j in range(i + 1)]

                n = len(confs)
                import sys
                sys.setrecursionlimit(100000)
                logger.info(
                    'Using %i cores for the 3-body energy-energy gram matrix calculation' % (ncores))

                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(ncores + 1)
                factor = (n + (ncores - 1)) / ncores
                splitind[1:-1] = [(i + 1) *
                                  factor for i in range(ncores - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [[confs[splitind[i]:splitind[i + 1]], self.theta[0], self.theta[1], self.theta[2],
                          self.type, False] for i in range(ncores)]  # Shape is ncores * (ntrain*(ntrain+1)/2)/ncores
                del confs
                import multiprocessing as mp
                pool = mp.Pool(ncores)
                result = pool.map(dummy_calc_ee, clist)
                del clist
                pool.close()
                pool.join()

                result = np.concatenate(result).ravel()
                off_diag = np.zeros((len(X), len(X)))
                diag = np.zeros((len(X), len(X)))
                for i in range(len(X)):
                    diag[i, i] = result[i + i * (i + 1) // 2]
                    for j in range(i):
                        off_diag[i, j] = result[j + i * (i + 1) // 2]
                del result

            else:
                diag = np.zeros((X.shape[0], X.shape[0]))
                off_diag = np.zeros((X.shape[0], X.shape[0]))
                for i in range(X.shape[0]):
                    for k, conf1 in enumerate(X[i]):
                        diag[i, i] += 1/9.0*self.k3_ee(conf1, conf1,
                                                 self.theta[0], self.theta[1], self.theta[2])
                        for conf2 in X[i][:k]:
                            # *2 here to speed up the loop
                            diag[i, i] += 1/9.0*2.0*self.k3_ee(
                                conf1, conf2, self.theta[0], self.theta[1], self.theta[2])
                    for j in range(i):
                        for conf1 in X[i]:
                            for conf2 in X[j]:
                                off_diag[i, j] += 1/9.0*self.k3_ee(
                                    conf1, conf2, self.theta[0], self.theta[1], self.theta[2])

            gram = diag + off_diag + off_diag.T
            del diag, off_diag            
            return gram

    def calc_gram_ef(self, X, X_glob, ncores=1, eval_gradient=False):
        """
        Calculate the energy-force gram matrix for a set of configurations X.
        This returns a non-symmetric matrix which is equal to the transpose of 
        the force-energy gram matrix.

        Args:
            X (list): list of N1 M1x5 arrays containing xyz coordinates and atomic species
            X_glob (list): list of N2 M2x5 arrays containing xyz coordinates and atomic species
            ncores (int): Number of CPU nodes to use for multiprocessing (default is 1)
            eval_gradient (bool): if True, evaluate the gradient of the gram matrix

        Returns:
            gram (matrix): N2 x N1*3 gram matrix of the vector-valued kernels 

       """
        gram = np.zeros((X_glob.shape[0], X.shape[0] * 3))

        if eval_gradient:
            raise NotImplementedError('ERROR: GRADIENT NOT IMPLEMENTED YET')
        else:
            if ncores > 1:  # Multiprocessing
                confs = [[x1, x2] for x1 in X_glob for x2 in X]
                n = len(confs)
                import sys
                sys.setrecursionlimit(100000)
                logger.info(
                    'Using %i cores for the 3-body energy-force gram matrix calculation' % (ncores))

                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(ncores + 1)
                factor = (n + (ncores - 1)) / ncores
                splitind[1:-1] = [(i + 1) *
                                  factor for i in range(ncores - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [[confs[splitind[i]:splitind[i + 1]], self.theta[0], self.theta[1], self.theta[2],
                          self.type, False] for i in range(ncores)]  # Shape is ncores * (ntrain*(ntrain+1)/2)/ncores

                del confs
                import multiprocessing as mp
                pool = mp.Pool(ncores)
                result = pool.map(dummy_calc_ef, clist)
                del clist
                pool.close()
                pool.join()

                result = np.concatenate(result).ravel()
                for i in range(X_glob.shape[0]):
                    for j in range(X.shape[0]):
                        gram[i, 3 * j:3 * j + 3] = result[3 *
                                                          (j + i * X.shape[0]):3 + 3*(j + i * X.shape[0])]
                del result
            else:
                for i in range(X_glob.shape[0]):
                    for j in range(X.shape[0]):
                        for k in X_glob[i]:
                            gram[i, 3 * j:3 * j + 3] += 1/3.0*self.k3_ef(
                                k, X[j], self.theta[0], self.theta[1], self.theta[2])

            return gram

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
        self.type = "single"

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
        if not (os.path.exists(Mffpath / 'k3_ee_s.pickle') and
                os.path.exists(Mffpath / 'k3_ef_s.pickle') and os.path.exists(Mffpath / 'k3_ff_s.pickle')):
            print("Building Kernels")

            import theano.tensor as T
            from theano import function, scan
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
            rjk = T.sqrt(
                T.sum((rho1s[None, :, :] - rho1s[:, None, :]) ** 2, axis=2))
            rmn = T.sqrt(
                T.sum((rho2s[None, :, :] - rho2s[:, None, :]) ** 2, axis=2))

            # --------------------------------------------------
            # BUILD THE KERNEL
            # --------------------------------------------------

            # Squared exp of differences
            se_1j2m = T.exp(-(r1j[:, None] - r2m[None, :])
                            ** 2 / (2 * sig ** 2))
            se_jkmn = T.exp(-(rjk[:, :, None, None] -
                              rmn[None, None, :, :]) ** 2 / (2 * sig ** 2))
            se_jk2m = T.exp(-(rjk[:, :, None] -
                              r2m[None, None, :]) ** 2 / (2 * sig ** 2))
            se_1jmn = T.exp(-(r1j[:, None, None] -
                              rmn[None, :, :]) ** 2 / (2 * sig ** 2))

            # Kernel not summed (cyclic permutations)
            k1n = (se_1j2m[:, None, :, None] *
                   se_1j2m[None, :, None, :] * se_jkmn)
            k2n = (se_1jmn[:, None, :, :] * se_jk2m[:, :,
                                                    None, :] * se_1j2m[None, :, :, None])
            k3n = (se_1j2m[:, None, None, :] *
                   se_jk2m[:, :, :, None] * se_1jmn[None, :, :, :])

            # final shape is M1 M1 M2 M2
            ker = k1n + k2n + k3n

            cut_j = 0.5*(1+T.cos(np.pi*r1j/rc))*((T.sgn(rc-r1j) + 1) / 2)
            cut_m = 0.5*(1+T.cos(np.pi*r2m/rc))*((T.sgn(rc-r2m) + 1) / 2)

            cut_jk = cut_j[:,None]*cut_j[None,:]*0.5*(1+T.cos(np.pi*rjk/rc))*((T.sgn(rc-rjk) + 1) / 2)
            cut_mn = cut_m[:,None]*cut_m[None,:]*0.5*(1+T.cos(np.pi*rmn/rc))*((T.sgn(rc-rmn) + 1) / 2)
            
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
            ker = T.sum(ker * cut_jk[:, :, None, None]
                        * cut_mn[None, None, :, :])

            # --------------------------------------------------
            # FINAL FUNCTIONS
            # --------------------------------------------------

            # global energy energy kernel
            k_ee_fun = function(
                [r1, r2, rho1, rho2, sig, theta, rc], ker, on_unused_input='ignore')

            # global energy force kernel
            k_ef = T.grad(ker, r2)
            k_ef_fun = function(
                [r1, r2, rho1, rho2, sig, theta, rc], k_ef, on_unused_input='ignore')

            # local force force kernel
            k_ff = T.grad(ker, r1)
            k_ff_der, updates = scan(lambda j, k_ff, r2: T.grad(k_ff[j], r2),
                                     sequences=T.arange(k_ff.shape[0]), non_sequences=[k_ff, r2])
            k_ff_fun = function(
                [r1, r2, rho1, rho2, sig, theta, rc], k_ff_der, on_unused_input='ignore')

            # Save the function that we want to use for multiprocessing
            # This is necessary because theano is a crybaby and does not want to access the
            # Automaticallly stored compiled object from different processes
            with open(Mffpath / 'k3_ee_s.pickle', 'wb') as f:
                pickle.dump(k_ee_fun, f)
            with open(Mffpath / 'k3_ef_s.pickle', 'wb') as f:
                pickle.dump(k_ef_fun, f)
            with open(Mffpath / 'k3_ff_s.pickle', 'wb') as f:
                pickle.dump(k_ff_fun, f)

        else:
            print("Loading Kernels")
            with open(Mffpath / "k3_ee_s.pickle", 'rb') as f:
                k_ee_fun = pickle.load(f)
            with open(Mffpath / "k3_ef_s.pickle", 'rb') as f:
                k_ef_fun = pickle.load(f)
            with open(Mffpath / "k3_ff_s.pickle", 'rb') as f:
                k_ff_fun = pickle.load(f)

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


class ThreeBodyManySpeciesKernel(BaseThreeBody):
    """Three body many species kernel.

    Args:
        theta[0] (float): lengthscale of the kernel
        theta[1] (float): decay rate of the cutoff function
        theta[2] (float): cutoff radius

    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        super().__init__(kernel_name='ThreeBodyManySpecies', theta=theta, bounds=bounds)
        self.type = "multi"

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

        if not (os.path.exists(Mffpath / 'k3_ee_m.pickle') and
                os.path.exists(Mffpath / 'k3_ef_m.pickle') and os.path.exists(Mffpath / 'k3_ff_m.pickle')):
            print("Building Kernels")

            import theano.tensor as T
            from theano import function, scan

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
            rjk = T.sqrt(
                T.sum((rho1s[None, :, :] - rho1s[:, None, :]) ** 2, axis=2))
            rmn = T.sqrt(
                T.sum((rho2s[None, :, :] - rho2s[:, None, :]) ** 2, axis=2))

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
            delta_alphas_jmkn = delta_alphasjm[:, None,
                                               :, None] * delta_alphasjm[None, :, None, :]

            delta_perm1 = delta_alphas12 * delta_alphas_jmkn

            # permutation 3
            delta_alphas1m = delta_alpha2(
                alpha_1[0, None], alpha_m[None, :]).flatten()
            delta_alphasjn = delta_alpha2(alpha_j[:, None], alpha_n[None, :])
            delta_alphask2 = delta_alpha2(
                alpha_k[:, None], alpha_2[None, 0]).flatten()

            delta_perm3 = delta_alphas1m[None, None, :, None] * delta_alphasjn[:, None, None, :] * \
                delta_alphask2[None, :, None, None]

            # permutation 5
            delta_alphas1n = delta_alpha2(
                alpha_1[0, None], alpha_n[None, :]).flatten()
            delta_alphasj2 = delta_alpha2(
                alpha_j[:, None], alpha_2[None, 0]).flatten()
            delta_alphaskm = delta_alpha2(alpha_k[:, None], alpha_m[None, :])

            delta_perm5 = delta_alphas1n[None, None, None, :] * delta_alphaskm[None, :, :, None] * \
                delta_alphasj2[:, None, None, None]

            # --------------------------------------------------
            # BUILD THE KERNEL
            # --------------------------------------------------

            # Squared exp of differences
            se_1j2m = T.exp(-(r1j[:, None] - r2m[None, :])
                            ** 2 / (2 * sig ** 2))
            se_jkmn = T.exp(-(rjk[:, :, None, None] -
                              rmn[None, None, :, :]) ** 2 / (2 * sig ** 2))
            se_jk2m = T.exp(-(rjk[:, :, None] -
                              r2m[None, None, :]) ** 2 / (2 * sig ** 2))
            se_1jmn = T.exp(-(r1j[:, None, None] -
                              rmn[None, :, :]) ** 2 / (2 * sig ** 2))

            # Kernel not summed (cyclic permutations)
            k1n = (se_1j2m[:, None, :, None] *
                   se_1j2m[None, :, None, :] * se_jkmn)
            k2n = (se_1jmn[:, None, :, :] * se_jk2m[:, :,
                                                    None, :] * se_1j2m[None, :, :, None])
            k3n = (se_1j2m[:, None, None, :] *
                   se_jk2m[:, :, :, None] * se_1jmn[None, :, :, :])

            # final shape is M1 M1 M2 M2

            ker_loc = k1n * delta_perm1 + k2n * delta_perm3 + k3n * delta_perm5

            # Faster version of cutoff (less calculations)
            cut_j = 0.5*(1+T.cos(np.pi*r1j/rc))#*((T.sgn(rc-r1j) + 1) / 2)
            cut_m = 0.5*(1+T.cos(np.pi*r2m/rc))#*((T.sgn(rc-r2m) + 1) / 2)

            cut_jk = cut_j[:,None]*cut_j[None,:]*0.5*(1+T.cos(np.pi*rjk/rc))*((T.sgn(rc-rjk) + 1) / 2)
            cut_mn = cut_m[:,None]*cut_m[None,:]*0.5*(1+T.cos(np.pi*rmn/rc))*((T.sgn(rc-rmn) + 1) / 2)


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
            ker_loc = T.sum(
                ker_loc * cut_jk[:, :, None, None] * cut_mn[None, None, :, :])

            # --------------------------------------------------
            # FINAL FUNCTIONS
            # --------------------------------------------------

            # energy energy kernel
            k_ee_fun = function(
                [r1, r2, rho1, rho2, sig, theta, rc], ker_loc, on_unused_input='ignore')

            # energy force kernel
            k_ef_cut = T.grad(ker_loc, r2)
            k_ef_fun = function(
                [r1, r2, rho1, rho2, sig, theta, rc], k_ef_cut, on_unused_input='ignore')

            # force force kernel
            k_ff_cut = T.grad(ker_loc, r1)
            k_ff_cut_der, updates = scan(lambda j, k_ff_cut, r2: T.grad(k_ff_cut[j], r2),
                                         sequences=T.arange(k_ff_cut.shape[0]), non_sequences=[k_ff_cut, r2])
            k_ff_fun = function(
                [r1, r2, rho1, rho2, sig, theta, rc], k_ff_cut_der, on_unused_input='ignore')

            # Save the function that we want to use for multiprocessing
            # This is necessary because theano is a crybaby and does not want to access the
            # Automaticallly stored compiled object from different processes
            with open(Mffpath / 'k3_ee_m.pickle', 'wb') as f:
                pickle.dump(k_ee_fun, f)
            with open(Mffpath / 'k3_ef_m.pickle', 'wb') as f:
                pickle.dump(k_ef_fun, f)
            with open(Mffpath / 'k3_ff_m.pickle', 'wb') as f:
                pickle.dump(k_ff_fun, f)

        else:
            print("Loading Kernels")
            with open(Mffpath / "k3_ee_m.pickle", 'rb') as f:
                k_ee_fun = pickle.load(f)
            with open(Mffpath / "k3_ef_m.pickle", 'rb') as f:
                k_ef_fun = pickle.load(f)
            with open(Mffpath / "k3_ff_m.pickle", 'rb') as f:
                k_ff_fun = pickle.load(f)

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
