import numpy as np
import _pickle as cPickle

import os

### Import Theano functions ###

theano_dir = os.path.dirname(os.path.abspath(__file__)) + '/theano_funcs/'

# three body
f = open(theano_dir + 'H3b_ms.save', 'rb')
threebody_ff_T = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + 'G3b_ms.save', 'rb')
threebody_ef_T = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + 'S3b_ms.save', 'rb')
threebody_ee_T = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + '3B_ff_cut.save', 'rb')
threebody_ff_T_cut = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + '3B_ef_cut.save', 'rb')
threebody_ef_T_cut = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + '3B_ee_cut.save', 'rb')
threebody_ee_T_cut = cPickle.load(f, encoding='latin1')
f.close()


# three body

def threebody_ff(a, b, sig, theta, rc):
    ret = threebody_ff_T(np.zeros(3), np.zeros(3), a, b, sig)
    return ret


def threebody_ef(a, b, sig, theta, rc):
    ret = threebody_ef_T(np.zeros(3), np.zeros(3), a, b, sig)
    return ret


def threebody_ee(a, b, sig, theta, rc):
    ret = threebody_ee_T(np.zeros(3), np.zeros(3), a, b, sig)
    return ret


def threebody_ff_cut(a, b, sig, rc, gamma):
    ret = threebody_ff_T_cut(np.zeros(3), np.zeros(3), a, b, sig, gamma, rc)
    return ret


def threebody_ef_cut(a, b, sig, rc, gamma):
    ret = threebody_ef_T_cut(np.zeros(3), np.zeros(3), a, b, sig, gamma, rc)
    return ret


def threebody_ee_cut(a, b, sig, rc, gamma):
    ret = threebody_ee_T_cut(np.zeros(3), np.zeros(3), a, b, sig, gamma, rc)
    return ret


# Classes for 2 and 3 body kernels
class TwoBody:
    """Two body kernel.
    
    Parameters
    ----------
    theta[0]: lengthscale
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
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


class ThreeBody:
    """Three body kernel.

    Parameters
    ----------
    theta[0]: lengthscale
    theta[1]: hardness of cutoff function (to be implemented)
    """

    def __init__(self, theta=(1., 1., 1.), bounds=((1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2))):
        self.theta = theta
        self.bounds = bounds

    def calc(self, X1, X2):

        K_trans = np.zeros((X1.shape[0] * 3, X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans[3 * i:3 * i + 3, 3 * j:3 * j + 3] = threebody_ff(X1[i], X2[j], self.theta[0],
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
                diag[3 * i:3 * i + 3, 3 * i:3 * i + 3] = threebody_ff(X[i], X[i], self.theta[0],
                                                                      self.theta[1], self.theta[2])
                for j in np.arange(i):
                    off_diag[3 * i:3 * i + 3, 3 * j:3 * j + 3] = threebody_ff(X[i], X[j], self.theta[0],
                                                                              self.theta[1], self.theta[2])

            gram = diag + off_diag + off_diag.T

            return gram

    def calc_diag(self, X):

        diag = np.zeros((X.shape[0] * 3))

        for i in np.arange(X.shape[0]):
            diag[i * 3:(i + 1) * 3] = np.diag(threebody_ff(X[i], X[i], self.theta[0], self.theta[1], self.theta[2]))

        return diag

    def calc_ef(self, X1, X2):

        K_trans = np.zeros((X1.shape[0], X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans[i, 3 * j:3 * j + 3] = threebody_ef(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

        return K_trans

    def calc_diag_e(self, X):

        diag = np.zeros((X.shape[0]))

        for i in np.arange(X.shape[0]):
            diag[i] = threebody_ee(X[i], X[i], self.theta[0], self.theta[1], self.theta[2])

        return diag
