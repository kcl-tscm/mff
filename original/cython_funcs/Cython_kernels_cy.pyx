import numpy as np
from cpython cimport array
cimport numpy as np
cimport cython

from libc.math cimport exp, sqrt

DTYPE = np.int
floatTYPE = np.float

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t

@cython.boundscheck(False)
@cython.cdivision(True)
def k2_calc_gram_ee(np.ndarray[floatTYPE_t, ndim = 2] confs, np.ndarray[DTYPE_t, ndim = 1] Ms , floatTYPE_t sig = .5):
	
	cdef int c1, c2, l1, l2, i, j

	cdef int n_confs = Ms.shape[0]

	cdef int M_sum1 = 0
	cdef int M_sum2 = 0

	cdef floatTYPE_t ker_c1c2 = 0.

	cdef floatTYPE_t d1, d2
	cdef floatTYPE_t twosigsq = 2* sig ** 2
	
	cdef np.ndarray[floatTYPE_t, ndim = 2] ker = np.zeros((n_confs, n_confs))
	cdef floatTYPE_t [:,:] ker_view = ker 

	cdef np.ndarray[floatTYPE_t, ndim = 1] dd = np.linalg.norm(confs, axis=1)
	
	for c1 in range(n_confs):
		for c2 in range(c1+1):  
		
			l1 = Ms[c1]
			l2 = Ms[c2]
		
			ker_c1c2 = 0.
		
			for i in range(l1):
				d1 = dd[M_sum1+i]
				for j in range(l2):
					d2 = dd[M_sum2+j]
					ker_c1c2 += exp(- (d1 - d2)**2 / twosigsq)
			
			ker_view[(c1):(c1+1), (c2):(c2+1)] = ker_c1c2
			M_sum2 += l2
			
		M_sum1 += l1
		M_sum2  = 0
		
	ker = ker + ker.T - np.diag(np.diag(ker))

	return ker
	

