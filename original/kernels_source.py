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

	print("Started compilation of theano kernels")

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
		d = T.exp(-(a1j - a2m) ** 2 / (2 * 1e-5 ** 2))
		return d

	delta_alphas12 = delta_alpha2(alpha_1[:, None], alpha_2[None, :])
	delta_alphasjm = delta_alpha2(alpha_j[:, None], alpha_m[None, :])
	delta_alphas1m = delta_alpha2(alpha_1[:, None], alpha_m[None, :])
	delta_alphasj2 = delta_alpha2(alpha_j[:, None], alpha_2[None, :])

	# Cutoff function
	k_ij = (T.exp(-(r1j[:, None] - r2m[None, :]) ** 2 / sig) * (
			delta_alphas12 * delta_alphasjm + delta_alphas1m * delta_alphasj2))

	# k_ij = k_ij * (T.exp(-theta / (rc - r1j[:, None])) * T.exp(-theta / (rc - r2m[None, :]))) / (
	#			T.exp(-theta / (rc - r1j[:, None])) * T.exp(-theta / (rc - r2m[None, :])))

	# kernel
	k = T.sum(k_ij)

	# FINAL FUNCTIONS

	# energy energy kernel
	k_ee_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k, allow_input_downcast=False, on_unused_input='warn')

	# energy force kernel
	k_ef = T.grad(k, r2)
	k_ef_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ef, allow_input_downcast=False, on_unused_input='warn')

	# force force kernel
	k_ff = T.grad(k, r1)
	k_ff_der, updates = scan(lambda j, k_ff, r2: T.grad(k_ff[j], r2),
	                         sequences=T.arange(k_ff.shape[0]), non_sequences=[k_ff, r2])

	k_ff_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ff_der, allow_input_downcast=False,
	                    on_unused_input='warn')

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

	print("Ended compilation of theano kernels")

	return k2_ee, k2_ef, k2_ff


def compile_threebody():
	"""
	This function generates theano compiled kernels for energy and force learning

	The position of the atoms relative to the centrla one, and their chemical species
	are defined by a matrix of dimension Mx5

	Returns:
		k3_ee (func): energy-energy kernel
		k3_ef (func): energy-force kernel
		k3_ff (func): force-force kernel
	"""

	print("Started compilation of theano kernels")

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
		d = np.exp(-(a1j - a2m) ** 2 / (2 * (1e-5) ** 2))
		return d

	# permutation 1

	delta_alphas12 = delta_alpha2(alpha_1[0], alpha_2[0])
	delta_alphasjm = delta_alpha2(alpha_j[:, None], alpha_m[None, :])
	delta_alphaskn = delta_alpha2(alpha_k[:, None], alpha_n[None, :])
	delta_alphas_jmkn = delta_alphasjm[:, None, :, None] * delta_alphaskn[None, :, None, :]

	delta_perm1 = delta_alphas12 * delta_alphas_jmkn

	# permutation 2

	delta_alphas12 = delta_alpha2(alpha_1[0], alpha_2[0])
	delta_alphasjn = delta_alpha2(alpha_j[:, None], alpha_n[None, :])
	delta_alphaskm = delta_alpha2(alpha_k[:, None], alpha_m[None, :])
	delta_alphas_jnkm = delta_alphasjn[:, None, :, None] * delta_alphaskm[None, :, None, :]

	delta_perm2 = delta_alphas12 * delta_alphas_jnkm

	# permutation 3
	delta_alphas1m = delta_alpha2(alpha_1[0, None], alpha_m[None, :]).flatten()
	delta_alphasjn = delta_alpha2(alpha_j[:, None], alpha_n[None, :])
	delta_alphask2 = delta_alpha2(alpha_k[:, None], alpha_2[None, 0]).flatten()
	delta_alphas_k21m = delta_alphask2[:, None] * delta_alphas1m[None, :]

	delta_perm3 = delta_alphas_k21m[:, None, :, None] * delta_alphasjn[None, :, None, :]

	# permutation 4
	delta_alphas1m = delta_alpha2(alpha_1[0, None], alpha_m[None, :]).flatten()
	delta_alphasj2 = delta_alpha2(alpha_j[:, None], alpha_2[None, 0]).flatten()
	delta_alphaskn = delta_alpha2(alpha_k[:, None], alpha_n[None, :])
	delta_alphas_j21m = delta_alphasj2[:, None] * delta_alphas1m[None, :]

	delta_perm4 = delta_alphas_j21m[:, None, :, None] * delta_alphaskn[None, :, None, :]

	# permutation 5
	delta_alphas1n = delta_alpha2(alpha_1[0, None], alpha_n[None, :]).flatten()
	delta_alphasj2 = delta_alpha2(alpha_j[:, None], alpha_2[None, 0]).flatten()
	delta_alphaskm = delta_alpha2(alpha_k[:, None], alpha_m[None, :])
	delta_alphas_j21n = delta_alphasj2[:, None] * delta_alphas1n[None, :]

	delta_perm5 = delta_alphas_j21n[:, None, :, None] * delta_alphaskm[None, :, None, :]

	# permutation 6
	delta_alphas1n = delta_alpha2(alpha_1[0, None], alpha_n[None, :]).flatten()
	delta_alphasjm = delta_alpha2(alpha_j[:, None], alpha_m[None, :])
	delta_alphask2 = delta_alpha2(alpha_k[:, None], alpha_2[None, 0]).flatten()
	delta_alphas_k21n = delta_alphask2[:, None] * delta_alphas1n[None, :]

	delta_perm6 = delta_alphas_k21n[:, None, :, None] * delta_alphasjm[None, :, None, :]

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

	ker_jkmn = (k1n + k2n + k3n) * (delta_perm1 + delta_perm2 + delta_perm3 + delta_perm4 + delta_perm5 + delta_perm6)

	cutoff_ikmn = (T.exp(-theta / (rc - r1j[:, None, None, None])) *
	               T.exp(-theta / (rc - r1j[None, :, None, None])) *
	               T.exp(-theta / (rc - rjk[:, :, None, None])) *
	               T.exp(-theta / (rc - r2m[None, None, None, :])) *
	               T.exp(-theta / (rc - r2m[None, None, :, None])) *
	               T.exp(-theta / (rc - rmn[None, None, :, :])) *
	               (0.5 * (T.sgn(rc - rjk[:, :, None, None]) + 1)) *
	               (0.5 * (T.sgn(rc - rmn[None, None, :, :]) + 1)))

	ker_jkmn_withcutoff = ker_jkmn * cutoff_ikmn

	# --------------------------------------------------
	# REMOVE DIAGONAL ELEMENTS
	# --------------------------------------------------

	mask_jk = T.ones_like(rjk) - T.identity_like(rjk)
	mask_mn = T.ones_like(rmn) - T.identity_like(rmn)

	mask_jkmn = mask_jk[:, :, None, None] * mask_mn[None, None, :, :]

	k_cutoff = T.sum(ker_jkmn_withcutoff * mask_jkmn)

	# --------------------------------------------------
	# FINAL FUNCTIONS
	# --------------------------------------------------

	# energy energy kernel
	k_ee_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_cutoff)

	# energy force kernel
	k_ef_cut = T.grad(k_cutoff, r2)
	k_ef_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ef_cut)

	# force force kernel
	k_ff_cut = T.grad(k_cutoff, r1)
	k_ff_cut_der, updates = theano.scan(lambda j, k_ff_cut, r2: T.grad(k_ff_cut[j], r2),
	                                    sequences=T.arange(k_ff_cut.shape[0]), non_sequences=[k_ff_cut, r2])
	k_ff_fun = function([r1, r2, rho1, rho2, sig, theta, rc], k_ff_cut_der)

	# WRAPPERS (we don't want to plug the position of the central element every time)

	def k3_ee(conf1, conf2, sig, theta, rc):
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

	def k3_ef(conf1, conf2, sig, theta, rc):
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

	def k3_ff(conf1, conf2, sig, theta, rc):
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

	print("Ended compilation of theano kernels")

	return k3_ee, k3_ef, k3_ff
