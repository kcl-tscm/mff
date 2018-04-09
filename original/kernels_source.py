import theano.tensor as T
from theano import function, scan
import numpy as np

def compile_twobody():
	"""
	This function generates theano compiled kernels for energy and force learning

	The position of the atoms relative to the centrla one, and their chemical species
	are defined by a matrix of dimension Mx5

	Returns:
		k2_ee (func): energy-energy kernel
		k2_ef (func): energy-force kernel
		k2_ff (func): force-force kernel
	"""
	# INITIAL DEFINITIONS

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

	# RELATIVE DISTANCES TO CENTRAL VECTOR

	# first and second configuration
	r1j = T.sqrt(T.sum((rho1s[:, :] - r1[None, :]) ** 2, axis=1))
	r2m = T.sqrt(T.sum((rho2s[:, :] - r2[None, :]) ** 2, axis=1))

	# CHEMICAL SPECIES MASK

	# numerical kronecker
	def delta_alpha2(a1j, a2m):
		d = T.exp(-(a1j - a2m) ** 2 / (2 * (1e-5) ** 2))
		return d

	delta_alphas12 = delta_alpha2(alpha_1[:, None], alpha_2[None, :])
	delta_alphasjm = delta_alpha2(alpha_j[:, None], alpha_m[None, :])
	delta_alphas1m = delta_alpha2(alpha_1[:, None], alpha_m[None, :])
	delta_alphasj2 = delta_alpha2(alpha_j[:, None], alpha_2[None, :])

	# Cutoff function
	k_ij = (T.exp(-(r1j[:, None] - r2m[None, :]) ** 2 / sig) * (
				delta_alphas12 * delta_alphasjm + delta_alphas1m * delta_alphasj2))

	k_ij = k_ij * (T.exp(-theta / (rc - r1j[:, None])) * T.exp(-theta / (rc - r2m[None, :]))) / (
				T.exp(-theta / (rc - r1j[:, None])) * T.exp(-theta / (rc - r2m[None, :])))

	# kernel
	k = T.sum(k_ij)

	# FINAL FUNCTIONS

	# energy energy kernel
	k_ee_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k, allow_input_downcast=True)

	# energy force kernel
	k_ef = T.grad(k, r2)
	k_ef_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ef, allow_input_downcast=True)

	# force force kernel
	k_ff = T.grad(k, r1)
	k_ff_der, updates = scan(lambda j, k_ff, r2: T.grad(k_ff[j], r2),
	                                sequences=T.arange(k_ff.shape[0]), non_sequences=[k_ff, r2])

	k_ff_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ff_der, allow_input_downcast=True)

	# WRAPPERS (we don't want to plug the position of the central element every time)

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
		return k_ee_fun(np.zeros(3),np.zeros(3), conf1, conf2, sig, theta, rc)

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

	return k2_ee, k2_ef, k2_ff


def compile_twobody():