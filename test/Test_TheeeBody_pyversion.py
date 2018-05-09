import numpy as np
import randrot as rd
import sys

sys.path.insert(0, '../')
from original import Kernels as K
import matplotlib.pyplot as plt

# ### Initial Definitions
k3 = K.ThreeBody(theta=[.5, 0.1, 100000.])


# define a function taking the positions
# of three atoms and creating a local configuration around the first

def build_conf(r1, r2, r3):
    r1, r2, r3 = np.array(r1, dtype=float), np.array(r2, dtype=float), np.array(r3, dtype=float)
    riv = r2[0:3] - r1[0:3]
    rjv = r3[0:3] - r1[0:3]

    ri = np.concatenate((riv, np.array([r1[3], r2[3]])))
    rj = np.concatenate((rjv, np.array([r1[3], r3[3]])))
    conf = np.reshape(np.array([[ri], [rj]]), (1, 2, 5))
    return conf


# test that three body works correctly
s1 = 1.
s2 = 1.
s3 = 100.

r1 = [0, 0, 0, s1]
r2 = [3, 0, 0, s2]
r3 = [0, 4, 0, s3]

c1 = build_conf(r1, r2, r3)
c2 = build_conf(r1, r3, r2)
c3 = build_conf(r2, r3, r1)
c4 = build_conf(r3, r2, r1)
print(k3.calc_ee(c1, c1))
print(k3.calc_ee(c1, c2))
print(k3.calc_ee(c1, c3))
print(k3.calc_ee(c1, c4))
print(k3.calc_ee(c2, c2))
print(k3.calc_ee(c3, c3))
print(k3.calc_ee(c4, c4))

# Define a random configuration
confR = np.random.normal(0, 1, (3, 3))
confR = np.hstack((confR, np.reshape([s1, s2, s3], (3, 1))))
confR_ = build_conf(confR[0], confR[1], confR[2])

k3.calc_ee(confR_, confR_)

confRP = confR[:]

confRP[0, :] = confR[2, :]
confRP[2, :] = confR[0, :]
confRP_ = build_conf(confRP[0], confRP[1], confRP[2])

print(k3.calc_ee(confR_, confRP_))

# ### Randomly sampled triplets
base_kernels = []

for i in np.arange(10):
    r1r = np.concatenate((np.random.normal(0, 1, (3,)), [s1]))
    r2r = np.concatenate((np.random.normal(0, 1, (3,)), [s2]))
    r3r = np.concatenate((np.random.normal(0, 1, (3,)), [s3]))
    c_ran = build_conf(r1r, r2r, r3r)
    base_kernels.append(k3.calc_ee(c_ran, c_ran)[0, 0])

print(base_kernels)

np.random.permutation(confR)

permuted_kernels = []

for i in np.arange(10):
    rs_per = np.random.permutation(confR)
    c_per = build_conf(rs_per[0], rs_per[1], rs_per[2])
    permuted_kernels.append(k3.calc_ee(c_per, c_per)[0, 0])

print(permuted_kernels)

# ### Random Rotations
rotated_kernels = []

for i in np.arange(10):
    RM = rd.generate(3)
    r1r, r2r, r3r = np.array(r1[:]), np.array(r2[:]), np.array(r3[:])
    r1r[0:3], r2r[0:3], r3r[0:3] = np.array(RM.dot(r1[0:3])), np.array(RM.dot(r2[0:3])), np.array(RM.dot(r3[0:3]))
    c_rot = build_conf(r1r, r2r, r3r)

    rotated_kernels.append(k3.calc_ee(c_rot, c_rot)[0, 0])

print(rotated_kernels)
