import logging
import numpy as np
from abc import ABCMeta, abstractmethod

from m_ff.kernels.base import Kernel

import theano.tensor as T
from theano import function, scan

logger = logging.getLogger(__name__)


class BaseTwoBody(Kernel, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, kernel_name, theta, bounds):
        super().__init__(kernel_name)
        self.theta = theta
        self.bounds = bounds

        self.k2_ee, self.k2_ef, self.k2_fe, self.k2_ff = self.compile_theano()

    def calc(self, X1, X2):

        K_trans = np.zeros((X1.shape[0] * 3, X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans[3 * i:3 * i + 3, 3 * j:3 * j + 3] = \
                    self.k2_ff(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

        return K_trans

    def calc_gram(self, X, eval_gradient=False):

        diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))
        off_diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))

        if eval_gradient:
            raise NotImplementedError('ERROR: GRADIENT NOT IMPLEMENTED YET')
        else:
            for i in np.arange(X.shape[0]):
                diag[3 * i:3 * i + 3, 3 * i:3 * i + 3] = \
                    self.k2_ff(X[i], X[i], self.theta[0], self.theta[1], self.theta[2])
                for j in np.arange(i):
                    off_diag[3 * i:3 * i + 3, 3 * j:3 * j + 3] = \
                        self.k2_ff(X[i], X[j], self.theta[0], self.theta[1], self.theta[2])

            gram = diag + off_diag + off_diag.T
            return gram

    def calc_diag(self, X):

        diag = np.zeros((X.shape[0] * 3))

        for i in np.arange(X.shape[0]):
            diag[i * 3:(i + 1) * 3] = np.diag(self.k2_ff(X[i], X[i], self.theta[0], self.theta[1], self.theta[2]))

        return diag

    def calc_ef(self, X1, X2):

        K_trans = np.zeros((X1.shape[0], X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans[i, 3 * j:3 * j + 3] = self.k2_ef(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

        return K_trans
    
    def calc_fe(self, X1, X2):

        K_trans_fe = np.zeros((X1.shape[0] * 3, X2.shape[0]))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans_fe[3 * i:3 * i + 3, j] = self.k2_fe(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

        return K_trans_fe

    def calc_diag_e(self, X):

        diag = np.zeros((X.shape[0]))

        for i in np.arange(X.shape[0]):
            diag[i] = self.k2_ee(X[i], X[i], self.theta[0], self.theta[1], self.theta[2])

        return diag

    def calc_gram_e(self, X, eval_gradient=False):

        diag = np.zeros((X.shape[0], X.shape[0]))
        off_diag = np.zeros((X.shape[0], X.shape[0]))

        if eval_gradient:
            raise NotImplementedError('ERROR: GRADIENT NOT IMPLEMENTED YET')
        else:
            for i in np.arange(X.shape[0]):
                diag[i,i] = \
                    self.k2_ee(X[i], X[i], self.theta[0], self.theta[1], self.theta[2])
                for j in np.arange(i):
                    off_diag[i,j] = \
                        self.k2_ee(X[i], X[j], self.theta[0], self.theta[1], self.theta[2])

            gram = diag + off_diag + off_diag.T
            return gram
        
    @staticmethod
    @abstractmethod
    def compile_theano():
        return None, None, None, None


class TwoBodySingleSpeciesKernel(BaseTwoBody):
    """Two body kernel.

    Parameters
    ----------
    theta[0]: lengthscale
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        super().__init__(kernel_name='TwoBodySingleSpecies', theta=theta, bounds=bounds)

    @staticmethod
    def compile_theano():
        """
        This function generates theano compiled kernels for energy and force learning

        The position of the atoms relative to the centrla one, and their chemical species
        are defined by a matrix of dimension Mx5

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
        
        # force energy kernel- Used to predict forces from energies
        k_fe = T.grad(k, r1)
        k_fe_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ef,
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
                conf1: first configuration.
                conf2: second configuration.
                sig: lengthscale hyperparameter.
                theta: cutoff smoothness hyperparameter.
                rc: cutoff distance hyperparameter.

            Returns:
                kernel (scalar):

            """
            return k_ee_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        def k2_ef(conf1, conf2, sig, theta, rc):
            """
            Two body kernel for energy-force correlation

            Args:
                conf1: first configuration.
                conf2: second configuration.
                sig: lengthscale hyperparameter.
                theta: cutoff smoothness hyperparameter.
                rc: cutoff distance hyperparameter.

            Returns:
                kernel (vector):
            """

            return k_ef_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        def k2_fe(conf1, conf2, sig, theta, rc):
            """
            Two body kernel for force-energy correlation

            Args:
                conf1: first configuration.
                conf2: second configuration.
                sig: lengthscale hyperparameter.
                theta: cutoff smoothness hyperparameter.
                rc: cutoff distance hyperparameter.

            Returns:
                kernel (vector):
            """

            return k_fe_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)
        
        
        def k2_ff(conf1, conf2, sig, theta, rc):
            """
            Two body kernel for force-force correlation

            Args:
                conf1: first configuration.
                conf2: second configuration.
                sig: lengthscale hyperparameter.
                theta: cutoff smoothness hyperparameter.
                rc: cutoff distance hyperparameter.

            Returns:
                kernel (matrix):
            """

            return k_ff_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        logger.info("Ended compilation of theano two body single species kernels")

        return k2_ee, k2_ef, k2_fe, k2_ff


class TwoBodyTwoSpeciesKernel(BaseTwoBody):
    """Two body kernel.

    Parameters
    ----------
    theta[0]: lengthscale
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        super().__init__(kernel_name='TwoBody', theta=theta, bounds=bounds)

    @staticmethod
    def compile_theano():
        """
        This function generates theano compiled kernels for energy and force learning

        The position of the atoms relative to the centrla one, and their chemical species
        are defined by a matrix of dimension Mx5

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

        # force energy kernel
        k_fe = T.grad(k, r1)
        k_fe_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ef,
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
                conf1: first configuration.
                conf2: second configuration.
                sig: lengthscale hyperparameter.
                theta: cutoff smoothness hyperparameter.
                rc: cutoff distance hyperparameter.

            Returns:
                kernel (scalar):

            """
            return k_ee_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        def k2_ef(conf1, conf2, sig, theta, rc):
            """
            Two body kernel for energy-force correlation

            Args:
                conf1: first configuration.
                conf2: second configuration.
                sig: lengthscale hyperparameter.
                theta: cutoff smoothness hyperparameter.
                rc: cutoff distance hyperparameter.

            Returns:
                kernel (vector):
            """

            return k_ef_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        def k2_fe(conf1, conf2, sig, theta, rc):
            """
            Two body kernel for force-energy correlation

            Args:
                conf1: first configuration.
                conf2: second configuration.
                sig: lengthscale hyperparameter.
                theta: cutoff smoothness hyperparameter.
                rc: cutoff distance hyperparameter.

            Returns:
                kernel (vector):
            """

            return k_fe_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)
        
        def k2_ff(conf1, conf2, sig, theta, rc):
            """
            Two body kernel for force-force correlation

            Args:
                conf1: first configuration.
                conf2: second configuration.
                sig: lengthscale hyperparameter.
                theta: cutoff smoothness hyperparameter.
                rc: cutoff distance hyperparameter.
            """

            return k_ff_fun(np.zeros(3), np.zeros(3), conf1, conf2, sig, theta, rc)

        logger.info("Ended compilation of theano two body kernels")

        return k2_ee, k2_ef, k2_fe, k2_ff
