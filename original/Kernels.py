import numpy as np


# Classes for 2 and 3 body kernels
class TwoBody:
    """Two body kernel.
    
    Parameters
    ----------
    theta[0]: lengthscale
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        self.kernel_name = 'TwoBody'
        self.theta = theta
        self.bounds = bounds

        from original.kernels_source import compile_twobody
        self.k2_ee, self.k2_ef, self.k2_ff = compile_twobody()

    def calc(self, X1, X2):

        K_trans = np.zeros((X1.shape[0] * 3, X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans[3 * i:3 * i + 3, 3 * j:3 * j + 3] = self.k2_ff(X1[i], X2[j], self.theta[0],
                                                                       self.theta[1], self.theta[2])
        return K_trans

    def calc_gram(self, X, eval_gradient=False):

        diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))
        off_diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))

        if eval_gradient:
            print('ERROR: GRADIENT NOT IMPLEMENTED YET')
            quit()

        else:
            for i in np.arange(X.shape[0]):
                diag[3 * i:3 * i + 3, 3 * i:3 * i + 3] = self.k2_ff(X[i], X[i], self.theta[0],
                                                                    self.theta[1], self.theta[2])
                for j in np.arange(i):
                    off_diag[3 * i:3 * i + 3, 3 * j:3 * j + 3] = self.k2_ff(X[i], X[j], self.theta[0],
                                                                            self.theta[1], self.theta[2])

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

    def calc_diag_e(self, X):

        diag = np.zeros((X.shape[0]))

        for i in np.arange(X.shape[0]):
            diag[i] = self.k2_ee(X[i], X[i], self.theta[0], self.theta[1], self.theta[2])

        return diag


class TwoBodySingleSpecies:
    """Two body kernel.

    Parameters
    ----------
    theta[0]: lengthscale
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        self.kernel_name = 'TwoBodySingleSpecies'
        self.theta = theta
        self.bounds = bounds

        from original.kernels_source import compile_twobody_singlespecies
        self.k2_ee, self.k2_ef, self.k2_ff = compile_twobody_singlespecies()

    def calc(self, X1, X2):

        K_trans = np.zeros((X1.shape[0] * 3, X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans[3 * i:3 * i + 3, 3 * j:3 * j + 3] = self.k2_ff(X1[i], X2[j], self.theta[0],
                                                                       self.theta[1], self.theta[2])
        return K_trans

    def calc_gram(self, X, eval_gradient=False):

        diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))
        off_diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))

        if eval_gradient:
            print('ERROR: GRADIENT NOT IMPLEMENTED YET')
            quit()

        else:
            for i in np.arange(X.shape[0]):
                diag[3 * i:3 * i + 3, 3 * i:3 * i + 3] = self.k2_ff(X[i], X[i], self.theta[0],
                                                                    self.theta[1], self.theta[2])
                for j in np.arange(i):
                    off_diag[3 * i:3 * i + 3, 3 * j:3 * j + 3] = self.k2_ff(X[i], X[j], self.theta[0],
                                                                            self.theta[1], self.theta[2])

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

    def calc_diag_e(self, X):

        diag = np.zeros((X.shape[0]))

        for i in np.arange(X.shape[0]):
            diag[i] = self.k2_ee(X[i], X[i], self.theta[0], self.theta[1], self.theta[2])

        return diag


class ThreeBody:
    """Three body kernel.

    Parameters
    ----------
    theta[0]: lengthscale
    theta[1]: hardness of cutoff function
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        self.kernel_name = 'ThreeBody'
        self.theta = theta
        self.bounds = bounds

        from original.kernels_source import compile_threebody
        self.k3_ee, self.k3_ef, self.k3_ff = compile_threebody()

    def calc(self, X1, X2):

        K_trans = np.zeros((X1.shape[0] * 3, X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans[3 * i:3 * i + 3, 3 * j:3 * j + 3] = self.k3_ff(X1[i], X2[j], self.theta[0],
                                                                       self.theta[1], self.theta[2])

        return K_trans

    def calc_ee(self, X1, X2):

        k = np.zeros((X1.shape[0], X2.shape[0]))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                k[i, j] = self.k3_ee(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

        return k

    def calc_ef(self, X1, X2):

        K_trans = np.zeros((X1.shape[0], X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans[i, 3 * j:3 * j + 3] = self.k3_ef(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

        return K_trans

    def calc_gram(self, X, eval_gradient=False):

        diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))
        off_diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))

        if eval_gradient:
            print('ERROR: GRADIENT NOT IMPLEMENTED YET')
            quit()
        else:

            for i in np.arange(X.shape[0]):
                diag[3 * i:3 * i + 3, 3 * i:3 * i + 3] = self.k3_ff(X[i], X[i], self.theta[0],
                                                                    self.theta[1], self.theta[2])
                for j in np.arange(i):
                    off_diag[3 * i:3 * i + 3, 3 * j:3 * j + 3] = self.k3_ff(X[i], X[j], self.theta[0],
                                                                            self.theta[1], self.theta[2])

            gram = diag + off_diag + off_diag.T

            return gram

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


class ThreeBodySingleSpecies:
    """Three body kernel.

    Parameters
    ----------
    theta[0]: lengthscale
    theta[1]: hardness of cutoff function
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        self.kernel_name = 'ThreeBodySingleSpecies'
        self.theta = theta
        self.bounds = bounds

        from original.kernels_source import compile_threebody_singlespecies
        self.k3_ee, self.k3_ef, self.k3_ff = compile_threebody_singlespecies()

    def calc(self, X1, X2):

        K_trans = np.zeros((X1.shape[0] * 3, X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans[3 * i:3 * i + 3, 3 * j:3 * j + 3] = self.k3_ff(X1[i], X2[j], self.theta[0],
                                                                       self.theta[1], self.theta[2])

        return K_trans

    def calc_gram(self, X, eval_gradient=False):

        diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))
        off_diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))

        if eval_gradient:
            print('ERROR: GRADIENT NOT IMPLEMENTED YET')
            quit()
        else:

            for i in np.arange(X.shape[0]):
                diag[3 * i:3 * i + 3, 3 * i:3 * i + 3] = self.k3_ff(X[i], X[i], self.theta[0],
                                                                    self.theta[1], self.theta[2])
                for j in np.arange(i):
                    off_diag[3 * i:3 * i + 3, 3 * j:3 * j + 3] = self.k3_ff(X[i], X[j], self.theta[0],
                                                                            self.theta[1], self.theta[2])

            gram = diag + off_diag + off_diag.T

            return gram

    def calc_diag(self, X):

        diag = np.zeros((X.shape[0] * 3))

        for i in np.arange(X.shape[0]):
            diag[i * 3:(i + 1) * 3] = np.diag(self.k3_ff(X[i], X[i], self.theta[0], self.theta[1], self.theta[2]))

        return diag

    def calc_ef(self, X1, X2):

        K_trans = np.zeros((X1.shape[0], X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans[i, 3 * j:3 * j + 3] = self.k3_ef(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

        return K_trans

    def calc_diag_e(self, X):

        diag = np.zeros((X.shape[0]))

        for i in np.arange(X.shape[0]):
            diag[i] = self.k3_ee(X[i], X[i], self.theta[0], self.theta[1], self.theta[2])

        return diag

    def calc_ee(self, X1, X2):

        k = np.zeros((X1.shape[0], X2.shape[0]))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                k[i, j] = self.k3_ee(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

        return k