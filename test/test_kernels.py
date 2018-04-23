import numpy as np
import randrot as rd


from original import Kernels as K

k3 = K.ThreeBody(theta=[1., 1000., 100000.])

# test that three body works correctly
s1 = 1.
s2 = 1.

s1v, s2v = np.array([s1, s1, s1]), np.array([s2, s2, s2])

# ----------------------------------------
# One chemical species, distance permutations
# ----------------------------------------

conf1 = np.array([[0.,1.,0., s1, s2],
				  [1.,0.,0., s1, s2]])

conf1 = np.reshape(conf1, (1, 2, 5))

print('kernel: \n', k3.calc_ee(conf1, conf1))

conf1P = np.zeros_like(conf1)

conf1P[0, 0, :], conf1P[0, 1, :] = conf1[0, 1, :], conf1[0, 0, :]

print('kernel permuted: \n', k3.calc_ee(conf1, conf1P))


# random rotation

for i in np.arange(10):
	RM = rd.generate(3)
	conf1R = np.empty_like(conf1)
	conf1R[:] = conf1#[:, :, :]

	for n in np.arange(conf1.shape[1]):

		aa = RM.dot(conf1[0, 0, 0:3])
		cc = aa[0].flatten()
		conf1R[0, 0, 0:3] = cc
		print(aa.shape)
		print(cc.shape)
		print(aa)
		print(cc)
		print(conf1R[0, 0, 0:3])

	print('kernel rotated: \n', k3.calc_ee(conf1, conf1R))

# ----------------------------------------
# Two chemical species, partial distance
# ----------------------------------------







# Test two body
if False:
	# Random configurations
	N = 10
	M = 5
	confs = np.random.normal(0, 0.5, (N, M, 5))
	confs[:, :, 3:5] = 1.

	theta_v = [1., 1., 1.]

	two = K.TwoBody(theta=theta_v)

	new = two.calc(confs, confs)

	print(new)

	rho1n = np.array([[[1, 2, 3, 1, 3],
	                   [1, 2, 4, 1, 4],
	                   [1, 2, 1, 1, 1]]])

	rho2n = np.array([[[1.2, 2.1, 3.3, 1, 3],
	                   [1.7, 2.1, 4.6, 1, 4],
	                   [1.2, 2.1, 1.2, 1, 1]]])

	print(two.calc(rho1n, rho2n))


# Test three body
if False:
	N = 10
	M = 5
	confs = np.random.normal(0, 0.5, (N, M, 5))
	confs[:, :, 3:5] = 1.

	theta = [1., 0.001, 10000.]

	two = K.ThreeBody(theta=theta)

	old = two.calc_old(confs, confs)

	new = two.calc(confs, confs)

	print(np.allclose(new,old))

	rho1n = np.array([[[1, 2, 3, 1, 3],
	                   [1, 2, 4, 1, 4],
	                   [1, 2, 1, 1, 1]]])

	rho2n = np.array([[[1.2, 2.1, 3.3, 1, 3],
	                   [1.7, 2.1, 4.6, 1, 4],
	                   [1.2, 2.1, 1.2, 1, 1]]])

	print(two.calc(rho1n, rho2n))
	print(two.calc_old(rho1n, rho2n))