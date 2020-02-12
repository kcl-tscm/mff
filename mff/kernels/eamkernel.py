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
        with open(Mffpath / "keam_ff_s.pickle", 'rb') as f:
            fun = pickle.load(f)
    elif kertype == "multi":
        with open(Mffpath / "keam_ff_m.pickle", 'rb') as f:
            fun = pickle.load(f)
    result = np.zeros((len(array), 3, 3))
    for i in np.arange(len(array)):
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

    array, theta0, theta1, theta2, kertype, mapping, alpha_1_descr = data
    if mapping:
        if kertype == "single":
            with open(Mffpath / "keam_eed_s.pickle", 'rb') as f:
                fun = pickle.load(f)
        elif kertype == "multi":
            with open(Mffpath / "keam_eed_m.pickle", 'rb') as f:
                fun = pickle.load(f)
    else:
        if kertype == "single":
            with open(Mffpath / "keam_ee_s.pickle", 'rb') as f:
                fun = pickle.load(f)
        elif kertype == "multi":
            with open(Mffpath / "keam_ee_m.pickle", 'rb') as f:
                fun = pickle.load(f)
    result = np.zeros(len(array))

    if not mapping:
        for i in np.arange(len(array)):
            for conf1 in array[i][0]:
                for conf2 in array[i][1]:
                    result[i] += 0.25*fun(np.zeros(3), np.zeros(3), conf1,
                                     conf2, theta0, theta1, theta2)
    else:
        if kertype == "multi":
            for i in np.arange(len(array)):
                for conf2 in array[i][1]:
                    result[i] += 0.5*fun(np.zeros(3), array[i]
                                     [0], conf2, theta0, theta1, theta2, alpha_1_descr)
        else:
            for i in np.arange(len(array)):
                for conf2 in array[i][1]:
                    result[i] += fun(np.zeros(3), array[i]
                                     [0], conf2, theta0, theta1, theta2)
    return result


def dummy_calc_ef(data):
    """ Function used when multiprocessing.
    Args:
        data (list of objects): contains all the information required
            for the computation of the kernel values
    
    Returns:
        result (array): the computed kernel values

    """
    
    array, theta0, theta1, theta2, kertype, mapping, alpha_1_descr = data
    if mapping:
        if kertype == "single":
            with open(Mffpath / "keam_efd_s.pickle", 'rb') as f:
                fun = pickle.load(f)
        elif kertype == "multi":
            with open(Mffpath / "keam_efd_m.pickle", 'rb') as f:
                fun = pickle.load(f)
    else:
        if kertype == "single":
            with open(Mffpath / "keam_ef_s.pickle", 'rb') as f:
                fun = pickle.load(f)
        elif kertype == "multi":
            with open(Mffpath / "keam_ef_m.pickle", 'rb') as f:
                fun = pickle.load(f)
    result = np.zeros((len(array), 3))
    if not mapping:
        for i in np.arange(len(array)):
            conf2 = np.array(array[i][1], dtype='float')
            for conf1 in array[i][0]:
                conf1 = np.array(conf1, dtype='float')
                result[i] += -0.5*fun(np.zeros(3), np.zeros(3), conf1,
                                  conf2,  theta0, theta1, theta2)
    else:
        if kertype == "multi":
            for i in np.arange(len(array)):
                conf2 = np.array(array[i][1], dtype='float')
                conf1 = np.array(array[i][0], dtype='float')
                result[i] += -fun(np.zeros(3), conf1,
                                  conf2, theta0, theta1, theta2, alpha_1_descr)
        else:
            for i in np.arange(len(array)):
                conf2 = np.array(array[i][1], dtype='float')
                conf1 = np.array(array[i][0], dtype='float')
                result[i] += -fun(np.zeros(3), conf1,
                                  conf2, theta0, theta1, theta2)
    return result


class BaseEam(Kernel, metaclass=ABCMeta):
    """ Eam kernel class
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
        self.k2_ee, self.k2_ef, self.k2_ff, self.k2_ee_d, self.k2_ef_d = self.compile_theano()

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
            confs = []
            for x1 in X1:
                for x2 in X2:
                    confs.append(np.asarray([x1, x2]))
            n = len(confs)
            import sys
            sys.setrecursionlimit(100000)
            logger.info(
                'Using %i cores for the eam force-force kernel calculation' % (ncores))

            # Way to split the kernels functions to compute evenly across the nodes
            splitind = np.zeros(ncores + 1)
            factor = (n + (ncores - 1)) / ncores
            splitind[1:-1] = [(i + 1) * factor for i in np.arange(ncores - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [[confs[splitind[i]:splitind[i + 1]], self.theta[0], self.theta[1], self.theta[2],
                      self.type] for i in np.arange(ncores)]  # Shape is ncores * (ntrain*(ntrain+1)/2)/ncores

            import multiprocessing as mp
            pool = mp.Pool(ncores)
            result = pool.map(dummy_calc_ff, clist)
            pool.close()
            pool.join()

            result = np.concatenate(result).reshape((n, 3, 3))
            for i in range(len(X1)):
                for j in range(len(X2)):
                    ker[i * 3: i * 3 + 3, 3 * j:3 * j +
                        3] = result[(j + i * len(X2))]

        else:
            for i, conf1 in enumerate(X1):
                for j, conf2 in enumerate(X2):
                    ker[i * 3:i * 3 + 3, 3 * j:3 * j + 3] += self.k2_ff(
                        conf1, conf2, self.theta[0], self.theta[1], self.theta[2])

        return ker

    def calc_ef(self, X_glob, X, ncores=1, mapping=False, alpha_1_descr=0):
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
            confs = []
            for x1 in X_glob:
                for x2 in X:
                    confs.append(np.asarray([x1, x2]))
            n = len(confs)
            import sys
            sys.setrecursionlimit(100000)
            logger.info(
                'Using %i cores for the eam energy-force kernel calculation' % (ncores))

            # Way to split the kernels functions to compute evenly across the nodes
            splitind = np.zeros(ncores + 1)
            factor = (n + (ncores - 1)) / ncores
            splitind[1:-1] = [(i + 1) * factor for i in np.arange(ncores - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [[confs[splitind[i]:splitind[i + 1]], self.theta[0], self.theta[1], self.theta[2],
                      self.type, mapping, alpha_1_descr] for i in np.arange(ncores)]  # Shape is ncores * (ntrain*(ntrain+1)/2)/ncores

            import multiprocessing as mp
            pool = mp.Pool(ncores)
            result = pool.map(dummy_calc_ef, clist)
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
                            ker[i, 3 * j:3 * j + 3] += 0.5*self.k2_ef(
                                conf1, conf2, self.theta[0], self.theta[1], self.theta[2])
            else:
                if self.type == 'multi':
                    for i, conf1 in enumerate(X_glob):
                            for j, conf2 in enumerate(X):
                                ker[i, 3 * j:3 * j + 3] += self.k2_ef_d(
                                    conf1, conf2, self.theta[0], self.theta[1], self.theta[2], alpha_1_descr)
                else:
                    for i, conf1 in enumerate(X_glob):
                        for j, conf2 in enumerate(X):
                            ker[i, 3 * j:3 * j + 3] += self.k2_ef_d(
                                conf1, conf2, self.theta[0], self.theta[1], self.theta[2])

        return ker

    def calc_ee(self, X1, X2, ncores=1, mapping=False, alpha_1_descr=0):
        """
        Calculate the energy-energy kernel between two global environments.

        Args:
            X1 (list): list of N1 Mx5 arrays containing xyz coordinates and atomic species
            X2 (list): list of N2 Mx5 arrays containing xyz coordinates and atomic species

        Returns:
            K (matrix): N1 x N2 matrix of the scalar-valued kernels 

       """
        if ncores > 1:  # Used for multiprocessing
            confs = []

            # Build a list of all input pairs which matrix needs to be computed
            for x1 in X1:
                for x2 in X2:
                    confs.append(np.asarray([x1, x2]))
            n = len(confs)
            import sys
            sys.setrecursionlimit(100000)
            logger.info(
                'Using %i cores for the eam energy-energy kernel calculation' % (ncores))

            # Way to split the kernels functions to compute evenly across the nodes
            splitind = np.zeros(ncores + 1)
            factor = (n + (ncores - 1)) / ncores
            splitind[1:-1] = [(i + 1) * factor for i in np.arange(ncores - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [[confs[splitind[i]:splitind[i + 1]], self.theta[0], self.theta[1], self.theta[2],
                      self.type, mapping, alpha_1_descr] for i in np.arange(ncores)]  # Shape is ncores * (ntrain*(ntrain+1)/2)/ncores

            import multiprocessing as mp
            pool = mp.Pool(ncores)
            result = pool.map(dummy_calc_ee, clist)
            pool.close()
            pool.join()
            result = np.concatenate(result).ravel()

            ker = np.zeros((len(X1), len(X2)))
            for i in range(len(X1)):
                for j in range(len(X2)):
                    ker[i, j] = result[j + i*len(X2)]

        else:
            if not mapping:
                ker = np.zeros((len(X1), len(X2)))
                for i, x1 in enumerate(X1):
                    for j, x2 in enumerate(X2):
                        for conf1 in x1:
                            for conf2 in x2:
                                ker[i, j] += 0.25*self.k2_ee(conf1, conf2, self.theta[0],
                                                        self.theta[1], self.theta[2])
            else:
                if self.type == 'multi':
                    ker = np.zeros((len(X1), len(X2)))
                    for i, conf1 in enumerate(X1):
                            for j, x2 in enumerate(X2):
                                for conf2 in x2:
                                    ker[i, j] += 0.5*self.k2_ee_d(
                                        conf1, conf2, self.theta[0], self.theta[1], self.theta[2], alpha_1_descr)
                else:
                    ker = np.zeros((len(X1), len(X2)))
                    for i, conf1 in enumerate(X1):
                        for j, x2 in enumerate(X2):
                            for conf2 in x2:
                                ker[i, j] += 0.5*self.k2_ee_d(
                                    conf1, conf2, self.theta[0], self.theta[1], self.theta[2])

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
            if ncores > 1:  # Used for multiprocessing
                confs = []

                # Build a list of all input pairs which matrix needs to be computed
                for i in np.arange(len(X)):
                    for j in np.arange(i + 1):
                        thislist = np.asarray([X[i], X[j]])
                        confs.append(thislist)
                n = len(confs)
                import sys
                sys.setrecursionlimit(100000)
                logger.info(
                    'Using %i cores for the eam force-force gram matrix calculation' % (ncores))

                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(ncores + 1)
                factor = (n + (ncores - 1)) / ncores
                splitind[1:-1] = [(i + 1) *
                                  factor for i in np.arange(ncores - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [[confs[splitind[i]:splitind[i + 1]], self.theta[0], self.theta[1], self.theta[2],
                          self.type] for i in np.arange(ncores)]  # Shape is ncores * (ntrain*(ntrain+1)/2)/ncores

                import multiprocessing as mp
                pool = mp.Pool(ncores)
                result = pool.map(dummy_calc_ff, clist)
                pool.close()
                pool.join()

                result = np.concatenate(result).reshape((n, 3, 3))
                off_diag = np.zeros((len(X) * 3, len(X) * 3))
                diag = np.zeros((len(X) * 3, len(X) * 3))
                for i in np.arange(len(X)):
                    diag[3 * i:3 * i + 3, 3 * i:3 * i +
                         3] = result[i + i * (i + 1) // 2]
                    for j in np.arange(i):
                        off_diag[3 * i:3 * i + 3, 3 * j:3 *
                                 j + 3] = result[j + i * (i + 1) // 2]

            else:
                diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))
                off_diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))
                for i in np.arange(X.shape[0]):
                    diag[3 * i:3 * i + 3, 3 * i:3 * i + 3] = \
                        self.k2_ff(X[i], X[i], self.theta[0],
                                   self.theta[1], self.theta[2])
                    for j in np.arange(i):
                        off_diag[3 * i:3 * i + 3, 3 * j:3 * j + 3] = \
                            self.k2_ff(X[i], X[j], self.theta[0],
                                       self.theta[1], self.theta[2])

            gram = diag + off_diag + off_diag.T  # The gram matrix is symmetric
            return gram

    def calc_gram_e(self, X, ncores=1, eval_gradient=False):
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
            if ncores > 1:  # Used for multiprocessing
                confs = []

                # Build a list of all input pairs which matrix needs to be computed
                for i in np.arange(len(X)):
                    for j in np.arange(i + 1):
                        thislist = np.array([list(X[i]), list(X[j])])
                        confs.append(thislist)

                n = len(confs)
                import sys
                sys.setrecursionlimit(100000)
                logger.info(
                    'Using %i cores for the eam energy-energy gram matrix calculation' % (ncores))

                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(ncores + 1)
                factor = (n + (ncores - 1)) / ncores
                splitind[1:-1] = [(i + 1) *
                                  factor for i in np.arange(ncores - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [[confs[splitind[i]:splitind[i + 1]], self.theta[0], self.theta[1], self.theta[2],
                          self.type, False, False] for i in np.arange(ncores)]  # Shape is ncores * (ntrain*(ntrain+1)/2)/ncores

                import multiprocessing as mp
                pool = mp.Pool(ncores)
                result = pool.map(dummy_calc_ee, clist)
                pool.close()
                pool.join()

                result = np.concatenate(result).ravel()
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
                        diag[i, i] += 0.25*self.k2_ee(conf1, conf1, self.theta[0],
                                                 self.theta[1], self.theta[2])
                        for conf2 in X[i][:k]:
                            # *2 here to speed up the loop
                            diag[i, i] += 0.25*2.0*self.k2_ee(
                                conf1, conf2, self.theta[0], self.theta[1], self.theta[2])
                    for j in np.arange(i):
                        for conf1 in X[i]:
                            for conf2 in X[j]:
                                off_diag[i, j] += 0.25*self.k2_ee(
                                    conf1, conf2, self.theta[0], self.theta[1], self.theta[2])

            gram = diag + off_diag + off_diag.T  # Gram matrix is symmetric

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
                confs = []
                for i in np.arange(len(X_glob)):
                    for j in np.arange(len(X)):
                        thislist = np.asarray([X_glob[i], X[j]])
                        confs.append(thislist)
                n = len(confs)
                import sys
                sys.setrecursionlimit(100000)
                logger.info(
                    'Using %i cores for the eam energy-force gram matrix calculation' % (ncores))

                # Way to split the kernels functions to compute evenly across the nodes
                splitind = np.zeros(ncores + 1)
                factor = (n + (ncores - 1)) / ncores
                splitind[1:-1] = [(i + 1) *
                                  factor for i in np.arange(ncores - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [[confs[splitind[i]:splitind[i + 1]], self.theta[0], self.theta[1], self.theta[2],
                          self.type, False, False] for i in np.arange(ncores)]  # Shape is ncores * (ntrain*(ntrain+1)/2)/ncores

                import multiprocessing as mp
                pool = mp.Pool(ncores)
                result = pool.map(dummy_calc_ef, clist)
                pool.close()
                pool.join()
                result = np.vstack(np.asarray(result))

                for i in np.arange(X_glob.shape[0]):
                    for j in np.arange(X.shape[0]):
                        gram[i, 3 * j:3 * j + 3] = result[(j + i * X.shape[0])]

            else:
                for i in np.arange(X_glob.shape[0]):
                    for j in np.arange(X.shape[0]):
                        for k in X_glob[i]:
                            gram[i, 3 * j:3 * j + 3] += 0.5*self.k2_ef(
                                k, X[j], self.theta[0], self.theta[1], self.theta[2])

            self.gram_ef = gram

            return gram

    @staticmethod
    @abstractmethod
    def compile_theano():
        return None, None, None, None, None


class EamSingleSpeciesKernel(BaseEam):
    """Eam single species kernel.

    Args:
        theta[0] (float): lengthscale of the kernel
        theta[1] (float): cutoff radius
        theta[2] (float): radius in the descriptor's exponent
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-1, 1e2), (1e-1, 1e2))):
        super().__init__(kernel_name='EamSingleSpecies', theta=theta, bounds=bounds)
        self.type = "single"

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
            k2_ee_map (func): energy-energy kernel that takes descriptor as one argument
            k2_ef_map (func): energy-force kernel that takes descriptor as one argument
        """

        if not (os.path.exists(Mffpath / 'keam_ee_s.pickle') and
                os.path.exists(Mffpath / 'keam_ef_s.pickle') and os.path.exists(
                    Mffpath / 'keam_ff_s.pickle')
                and os.path.exists(Mffpath / 'keam_eed_s.pickle') and os.path.exists(Mffpath / 'keam_efd_s.pickle')):
            print("Building Kernels")

            import theano.tensor as T
            from theano import function, scan
            logger.info(
                "Started compilation of theano eam single species kernels")
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
            rc = T.dscalar('rc')
            # Descriptor as a given input, used to map
            q1_descr = T.dscalar('q1_descr')
            # Radius to use at denominator in the descriptor
            r0 = T.dscalar('r0')

            # positions of neighbours without chemical species (3D space assumed)
            rho1s = rho1[:, 0:3]
            rho2s = rho2[:, 0:3]

            # distances of atoms wrt to the central one and wrt each other in 1 and 2
            r1j = T.sqrt(T.sum((rho1s[:, :] - r1[None, :]) ** 2, axis=1))
            r2m = T.sqrt(T.sum((rho2s[:, :] - r2[None, :]) ** 2, axis=1))

            esp_term_1 = (r1j/r0 - 1)
            esp_term_2 = (r2m/r0 - 1)

            cut_1 = 0.5*(1 + T.cos(np.pi*r1j/rc))*((T.sgn(rc-r1j) + 1) / 2)
            cut_2 = 0.5*(1 + T.cos(np.pi*r2m/rc))*((T.sgn(rc-r2m) + 1) / 2)

            q1 = T.sum(T.exp(-esp_term_1)*cut_1)
            q2 = T.sum(T.exp(-esp_term_2)*cut_2)

            k = T.exp(-(q1-q2)**2/(2*sig**2))

            k_descr = T.exp(-(q1_descr-q2)**2/(2*sig**2))

            # energy energy kernel
            k_ee_fun = function([r1, r2, rho1, rho2, sig, rc, r0], k,
                                allow_input_downcast=False, on_unused_input='warn')

            # energy force kernel - Used to predict energies from forces
            k_ef = T.grad(k, r2)
            k_ef_fun = function([r1, r2, rho1, rho2, sig, rc, r0], k_ef,
                                allow_input_downcast=False, on_unused_input='warn')

            # force force kernel - it uses only local atom pairs to avoid useless computation
            k_ff = T.grad(k, r1)
            k_ff_der, updates = scan(lambda j, k_ff, r2: T.grad(k_ff[j], r2),
                                     sequences=T.arange(k_ff.shape[0]), non_sequences=[k_ff, r2])

            k_ff_fun = function([r1, r2, rho1, rho2, sig, rc, r0], k_ff_der,
                                allow_input_downcast=False, on_unused_input='warn')

            # energy energy descriptor kernel
            k_ee_fun_d = function([r2, q1_descr, rho2, sig, rc, r0], k_descr,
                                  allow_input_downcast=False, on_unused_input='warn')

            # energy force descriptor kernel
            k_ef_descr = T.grad(k_descr, r2)
            k_ef_fun_d = function([r2, q1_descr, rho2, sig, rc, r0], k_ef_descr,
                                  allow_input_downcast=False, on_unused_input='warn')

            # Save the function that we want to use for multiprocessing
            # This is necessary because theano is a crybaby and does not want to access the
            # Automaticallly stored compiled object from different processes
            with open(Mffpath / 'keam_ee_s.pickle', 'wb') as f:
                pickle.dump(k_ee_fun, f)
            with open(Mffpath / 'keam_ef_s.pickle', 'wb') as f:
                pickle.dump(k_ef_fun, f)
            with open(Mffpath / 'keam_ff_s.pickle', 'wb') as f:
                pickle.dump(k_ff_fun, f)
            with open(Mffpath / 'keam_eed_s.pickle', 'wb') as f:
                pickle.dump(k_ee_fun_d, f)
            with open(Mffpath / 'keam_efd_s.pickle', 'wb') as f:
                pickle.dump(k_ef_fun_d, f)

        else:
            print("Loading Kernels")
            with open(Mffpath / "keam_ee_s.pickle", 'rb') as f:
                k_ee_fun = pickle.load(f)
            with open(Mffpath / "keam_ef_s.pickle", 'rb') as f:
                k_ef_fun = pickle.load(f)
            with open(Mffpath / "keam_ff_s.pickle", 'rb') as f:
                k_ff_fun = pickle.load(f)
            with open(Mffpath / 'keam_eed_s.pickle', 'rb') as f:
                k_ee_fun_d = pickle.load(f)
            with open(Mffpath / 'keam_efd_s.pickle', 'rb') as f:
                k_ef_fun_d = pickle.load(f)
        # --------------------------------------------------
        # WRAPPERS (we don't want to plug the position of the central element every time)
        # --------------------------------------------------

        def k2_ee(conf1, conf2, sig, rc, r0):
            """
            Eam kernel for global energy-energy correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                rc (float): cutoff distance hyperparameter theta[1]

            Returns:
                kernel (float): scalar valued energy-energy Eam kernel

            """
            return k_ee_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, rc, r0)

        def k2_ef(conf1, conf2, sig, rc, r0):
            """
            Eam kernel for global energy-force correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                rc (float): cutoff distance hyperparameter theta[1]

            Returns:
                kernel (array): 3x1 energy-force Eam kernel

            """
            return -k_ef_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, rc, r0)

        def k2_ff(conf1, conf2, sig, rc, r0):
            """
            Eam kernel for force-force correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                rc (float): cutoff distance hyperparameter theta[1]

            Returns:
                kernel (matrix): 3x3 force-force Eam kernel

            """
            return k_ff_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, rc, r0)

        def k2_ee_d(descr1, conf2,  sig, rc, r0):
            """
            Eam kernel for global energy-force correlation

            Args:
                descr1 (float): descriptor calculated for the first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                rc (float): cutoff distance hyperparameter theta[1]

            Returns:
                kernel (array): 3x1 energy-force Eam kernel

            """
            return k_ee_fun_d(np.zeros(3), descr1, conf2, sig, rc, r0)

        def k2_ef_d(descr1, conf2, sig, rc, r0):
            """
            Eam kernel for force-force correlation

            Args:
                descr1 (float): descriptor calculated for the first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                rc (float): cutoff distance hyperparameter theta[`]

            Returns:
                kernel (matrix): 3x3 force-force Eam kernel

            """
            return -k_ef_fun_d(np.zeros(3), descr1, conf2, sig, rc, r0)

        logger.info("Ended compilation of theano eam single species kernels")

        return k2_ee, k2_ef, k2_ff, k2_ee_d, k2_ef_d


class EamManySpeciesKernel(BaseEam):
    """Eam multi species kernel.

    Args:
        theta[0] (float): lengthscale of the kernel
        theta[1] (float): cutoff radius
        theta[2] (float): radius in the descriptor's exponent
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        super().__init__(kernel_name='EamMultiSpecies', theta=theta, bounds=bounds)
        self.type = "multi"

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
            k2_ee_map (func): energy-energy kernel that takes descriptor as one argument
            k2_ef_map (func): energy-force kernel that takes descriptor as one argument
        """

        if not (os.path.exists(Mffpath / 'keam_ee_m.pickle') and
                os.path.exists(Mffpath / 'keam_ef_m.pickle') and os.path.exists(
                    Mffpath / 'keam_ff_m.pickle')
                and os.path.exists(Mffpath / 'keam_eed_m.pickle') and os.path.exists(Mffpath / 'keam_efd_m.pickle')):
            print("Building Kernels")

            import theano.tensor as T
            from theano import function, scan
            logger.info(
                "Started compilation of theano eam multi species kernels")
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
            rc = T.dscalar('rc')
            # Descriptor as a given input, used to map
            q1_descr = T.dscalar('q1_descr')
            # Element of the central atom if descriptor is Given
            alpha_1_descr = T.dscalar('alpha_1_descr')
            # Radius to use at denominator in the descriptor
            r0 = T.dscalar('r0')

            # positions of neighbours without chemical species (3D space assumed)
            rho1s = rho1[:, 0:3]
            rho2s = rho2[:, 0:3]
            alpha_1 = rho1[0, 3]  # .flatten()
            alpha_2 = rho2[0, 3]  # .flatten()

            # numerical kronecker
            def delta_alpha(a1j, a2m):
                d = T.exp(-(a1j - a2m) ** 2 / (2 * 1e-5 ** 2))
                return d

            # matrices determining whether couples of atoms have the same atomic number
            delta_alpha_12 = delta_alpha(alpha_1, alpha_2)
            delta_alpha_12_descr = delta_alpha(alpha_1_descr, alpha_2)

            # distances of atoms wrt to the central one and wrt each other in 1 and 2
            r1j = T.sqrt(T.sum((rho1s[:, :] - r1[None, :]) ** 2, axis=1))
            r2m = T.sqrt(T.sum((rho2s[:, :] - r2[None, :]) ** 2, axis=1))

            esp_term_1 = (r1j/r0 - 1)
            esp_term_2 = (r2m/r0 - 1)

            cut_1 = 0.5*(1 + T.cos(np.pi*r1j/rc))*((T.sgn(rc-r1j) + 1) / 2)
            cut_2 = 0.5*(1 + T.cos(np.pi*r2m/rc))*((T.sgn(rc-r2m) + 1) / 2)

            q1 = T.sum(T.exp(-esp_term_1)*cut_1)
            q2 = T.sum(T.exp(-esp_term_2)*cut_2)

            k = T.exp(-(q1-q2)**2/(2*sig**2))*delta_alpha_12

            k_descr = T.exp(-(q1_descr-q2)**2/(2*sig**2))*delta_alpha_12_descr

            # energy energy kernel
            k_ee_fun = function([r1, r2, rho1, rho2, sig, rc, r0], k,
                                allow_input_downcast=False, on_unused_input='warn')

            # energy force kernel - Used to predict energies from forces
            k_ef = T.grad(k, r2)
            k_ef_fun = function([r1, r2, rho1, rho2, sig, rc, r0], k_ef,
                                allow_input_downcast=False, on_unused_input='warn')

            # force force kernel - it uses only local atom pairs to avoid useless computation
            k_ff = T.grad(k, r1)
            k_ff_der, updates = scan(lambda j, k_ff, r2: T.grad(k_ff[j], r2),
                                     sequences=T.arange(k_ff.shape[0]), non_sequences=[k_ff, r2])

            k_ff_fun = function([r1, r2, rho1, rho2, sig, rc, r0], k_ff_der,
                                allow_input_downcast=False, on_unused_input='warn')

            # energy energy descriptor kernel
            k_ee_fun_d = function([r2, q1_descr, rho2, sig, rc, r0, alpha_1_descr], k_descr,
                                  allow_input_downcast=False, on_unused_input='warn')

            # energy force descriptor kernel
            k_ef_descr = T.grad(k_descr, r2)
            k_ef_fun_d = function([r2, q1_descr, rho2, sig, rc, r0, alpha_1_descr], k_ef_descr,
                                  allow_input_downcast=False, on_unused_input='warn')

            # Save the function that we want to use for multiprocessing
            # This is necessary because theano is a crybaby and does not want to access the
            # Automaticallly stored compiled object from different processes
            with open(Mffpath / 'keam_ee_m.pickle', 'wb') as f:
                pickle.dump(k_ee_fun, f)
            with open(Mffpath / 'keam_ef_m.pickle', 'wb') as f:
                pickle.dump(k_ef_fun, f)
            with open(Mffpath / 'keam_ff_m.pickle', 'wb') as f:
                pickle.dump(k_ff_fun, f)
            with open(Mffpath / 'keam_eed_m.pickle', 'wb') as f:
                pickle.dump(k_ee_fun_d, f)
            with open(Mffpath / 'keam_efd_m.pickle', 'wb') as f:
                pickle.dump(k_ef_fun_d, f)

        else:
            print("Loading Kernels")
            with open(Mffpath / "keam_ee_m.pickle", 'rb') as f:
                k_ee_fun = pickle.load(f)
            with open(Mffpath / "keam_ef_m.pickle", 'rb') as f:
                k_ef_fun = pickle.load(f)
            with open(Mffpath / "keam_ff_m.pickle", 'rb') as f:
                k_ff_fun = pickle.load(f)
            with open(Mffpath / 'keam_eed_m.pickle', 'rb') as f:
                k_ee_fun_d = pickle.load(f)
            with open(Mffpath / 'keam_efd_m.pickle', 'rb') as f:
                k_ef_fun_d = pickle.load(f)
        # --------------------------------------------------
        # WRAPPERS (we don't want to plug the position of the central element every time)
        # --------------------------------------------------

        def k2_ee(conf1, conf2, sig, rc, r0):
            """
            Eam kernel for global energy-energy correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                rc (float): cutoff distance hyperparameter theta[1]

            Returns:
                kernel (float): scalar valued energy-energy Eam kernel

            """
            return k_ee_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, rc, r0)

        def k2_ef(conf1, conf2, sig, rc, r0):
            """
            Eam kernel for global energy-force correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                rc (float): cutoff distance hyperparameter theta[1]

            Returns:
                kernel (array): 3x1 energy-force Eam kernel

            """
            return -k_ef_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, rc, r0)

        def k2_ff(conf1, conf2, sig, rc, r0):
            """
            Eam kernel for force-force correlation

            Args:
                conf1 (array): first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                rc (float): cutoff distance hyperparameter theta[1]

            Returns:
                kernel (matrix): 3x3 force-force Eam kernel

            """
            return k_ff_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, rc, r0)

        def k2_ee_d(descr1, conf2,  sig, rc, r0, alpha_1_descr):
            """
            Eam kernel for global energy-force correlation

            Args:
                descr1 (float): descriptor calculated for the first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                rc (float): cutoff distance hyperparameter theta[2]
                alpha_1_descr (int): element of the central atom

            Returns:
                kernel (array): 3x1 energy-force Eam kernel

            """
            return k_ee_fun_d(np.zeros(3), descr1, conf2, sig, rc, r0, alpha_1_descr)

        def k2_ef_d(descr1, conf2, sig, rc, r0, alpha_1_descr):
            """
            Eam kernel for force-force correlation

            Args:
                descr1 (float): descriptor calculated for the first configuration.
                conf2 (array): second configuration.
                sig (float): lengthscale hyperparameter theta[0]
                rc (float): cutoff distance hyperparameter theta[1]
                alpha_1_descr (int): element of the central atom

            Returns:
                kernel (matrix): 3x3 force-force Eam kernel

            """
            return -k_ef_fun_d(np.zeros(3), descr1, conf2, sig, rc, r0, alpha_1_descr)

        logger.info("Ended compilation of theano eam multi species kernels")

        return k2_ee, k2_ef, k2_ff, k2_ee_d, k2_ef_d
