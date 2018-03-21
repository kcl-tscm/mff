from six.moves import cPickle
import sys

import theano
import theano.tensor as T
import theano.typed_list as Tl
from theano import function
import numpy as np

############ Definition of all variables ############

r1, r2 = T.dvectors('r1d','r2d')            # positions of central atoms
rho1, rho2 = T.dmatrices('rho1','rho2')     # positions of neighbours 
sig = T.dscalar('sig')                      # lenghtsclae hyperparameter
theta = T.dscalar('theta')                  # cutoff hardness hyperparameter
rc = T.dscalar('rc')                        # cutoff hyperparameter

rho1l = Tl.TypedListType(T.dmatrix)('rho1l')# first list of configurations
rho2l = Tl.TypedListType(T.dmatrix)('rho2l')# second list of configurations

l1, l2 = Tl.length(rho1l), Tl.length(rho2l)
range1, range2 = T.arange(l1, dtype='int64'), T.arange(l2, dtype='int64')


# numerical kronecker
def delta_alpha2(a1j, a2m):
    
    d = np.exp(-(a1j - a2m)**2/(2*(1e-5)**2))
    #d = np.exp((a1j[0] - a2m[0])**2/(2*(1e-7)**2))
    #dp = np.exp((a1j[1] - a2m[1])**2/(2*(1e-7)**2))
    
    return d

############ Definition of two body kernel and its derivatives ############

def kernel_ee_(r1, r2, rho1, rho2, sig, theta, rc):
    rho1s = rho1[:, 0:3]
    rho2s = rho2[:, 0:3]
    alpha_1 = rho1[:, 3:4].flatten()
    alpha_2 = rho2[:, 3:4].flatten()
    alpha_j = rho1[:, 4:5].flatten()
    alpha_m = rho2[:, 4:5].flatten()
    r1j = T.sqrt(T.sum((rho1s[:, :]-r1[None,:])**2, axis = 1))
    r2m = T.sqrt(T.sum((rho2s[:, :]-r2[None,:])**2, axis = 1))
    delta_alphas12 = delta_alpha2(alpha_1[:, None] , alpha_2[None, :])
    delta_alphasjm = delta_alpha2(alpha_j[:, None] , alpha_m[None, :])
    delta_alphas1m = delta_alpha2(alpha_1[:, None] , alpha_m[None, :])
    delta_alphasj2 = delta_alpha2(alpha_j[:, None] , alpha_2[None, :])
    k_ij = (T.exp(-(r1j[:, None] - r2m[None,:])**2/sig) * (delta_alphas12 * delta_alphasjm + delta_alphas1m * delta_alphasj2)
    *T.exp(-theta/(rc-r1j[:, None]))*T.exp(-theta/(rc-r2m[None,:])))
    k = T.sum(k_ij) 
    return k

def kernel_ef_(r1, r2, rho1, rho2, sig, theta, rc):
    k = kernel_ee_(r1, r2, rho1, rho2, sig, theta, rc)
    k_ef = T.grad(k, r2)

    return k_ef


def kernel_ff_(r1, r2, rho1, rho2, sig, theta, rc):
    k = kernel_ee_(r1, r2, rho1, rho2, sig, theta, rc)
    k_ff_der = T.grad(k, r1)
    k_ff, updates = theano.scan(lambda j, k_ff, r2 : T.grad(k_ff_der[j], r2), 
                                sequences = T.arange(k_ff_der.shape[0]), non_sequences = [k_ff_der, r2])

    return k_ff
    
############ Definition of functions to loop over ############   
        
def fun_ee(j, e1, list2, r1, r2, sig, theta, rc):
    return kernel_ee_(r1, r2, e1, list2[j], sig, theta, rc)

def fun_ff(j, e1, list2, r1, r2, sig, theta, rc):
    return kernel_ff_(r1, r2, e1, list2[j], sig, theta, rc)

def fun_ef(j, e1, list2, r1, r2, sig, theta, rc):
    return kernel_ef_(r1, r2, e1, list2[j], sig, theta, rc)


def loop_over_list_ee(i, range2, list1, list2, r1, r2, sig, theta, rc):
    e1 = list1[i]
    result_, updates_ = theano.scan(fn=fun_ee,
                              outputs_info = None,
                              non_sequences=[e1, list2, r1, r2,  sig, theta, rc],
                              sequences=[range2])
    return result_

def loop_over_list_ef(i, range2, list1, list2, r1, r2, sig, theta, rc):
    e1 = list1[i]
    result_, updates_ = theano.scan(fn=fun_ef,
                              outputs_info = None,
                              non_sequences=[e1, list2, r1, r2, sig, theta, rc],
                              sequences=[range2])
    return result_


def loop_over_list_ff(i, range2, list1, list2, r1, r2, sig, theta, rc):
    e1 = list1[i]
    result_, updates_ = theano.scan(fn=fun_ff,
                              outputs_info = None,
                              non_sequences=[e1, list2, r1, r2, sig, theta, rc],
                              sequences=[range2])
    return result_
    
############ Definition of kernel functionf over lists of configurations ############      
    
result, updates = theano.scan(fn=loop_over_list_ee,
                              outputs_info = None,
                              non_sequences=[range2, rho1l, rho2l, r1, r2, sig, theta, rc],
                              sequences=[range1])

k_ee_func_for = function(inputs=[rho1l, rho2l, r1, r2, sig, theta, rc], outputs=result)

result, updates = theano.scan(fn=loop_over_list_ef,
                              outputs_info = None,
                              non_sequences=[range2, rho1l, rho2l, r1, r2, sig, theta, rc],
                              sequences=[range1])

k_ef_func_for = function(inputs=[rho1l, rho2l, r1, r2, sig, theta, rc], outputs=result)

result, updates = theano.scan(fn=loop_over_list_ff,
                              outputs_info = None,
                              non_sequences=[range2, rho1l, rho2l, r1, r2, sig, theta, rc],
                              sequences=[range1])

k_ff_func_for = function(inputs=[rho1l, rho2l, r1, r2, sig, theta, rc], outputs=result)


