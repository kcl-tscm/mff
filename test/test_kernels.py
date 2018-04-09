import numpy as np
from original import Kernels as K

# Random configurations

N = 10
M = 5

confs = np.random.normal(0, 0.5, (N, M, 5))
confs[:, :, 3:5] = 1.

# Test two body

theta_v = [1., 1., 1.]

two = K.TwoBody(theta = theta_v)

new = two.calc(confs, confs)

print(new)

rho1n = np.array([[[1, 2, 3, 1, 3],
                  [1, 2, 4, 1, 4],
                  [1, 2, 1, 1, 1]]])

rho2n = np.array([[[1.2, 2.1, 3.3, 1, 3],
                  [1.7, 2.1, 4.6, 1, 4],
                  [1.2, 2.1, 1.2, 1, 1]]])

print(two.calc(rho1n, rho2n))
