Advanced Sampling
=================

This module contains some functions that can be used in order to subsample from very large datasets.


Theory/Introduction
-------------------


Example
-------

Assuming we already extracted all of the configurations, forces (and possibly local energies) from a .xyz file, we can apply one of the methods contained in advanced_sampling in order to subsample a meaningful and representative training set.

We first load the configurations and forces previously extracted from the .xyz file::

 confs = np.load(configurations_file)
 forces = np.load(configurations_file)
 
We then initialize the sampling class and separate ntest configurations for the test set::
 s = Sampling(confs=confs,forces=forces, sigma_2b = 0.05, sigma_3b = 0.1, sigma_mb = 0.2, noise = 0.001, r_cut = 8.5, theta = 0.5)
 s.train_test_split(confs=confs, forces = forces, ntest = 200)
 
Now we can subsample a training set using our preferred method, for example importance vector machine sampling on the variance of force predicion::
 MAE, STD, RMSE, index, time = s.ivm_f(method = '2b', ntrain = ntr, batchsize = 1000)

or importance vector machine sampling on the measured error of force predicion for a 3-body kernel::
 MAE, STD, RMSE, index, time = s.ivm_f(method = '3b', ntrain = ntr, batchsize = 1000, use_pred_error = False)
 
Other methods include a sampling based on the interatomic distance values present in every configuration::
 MAE, STD, RMSE, index, time = s.grid(method = '2b', nbins = 1000)
 
Or  a sampling based on the interatomic distance values present in every configuration::
 MAE, STD, RMSE, index, time = s.grid(method = '2b', nbins = 1000)

.. automodule:: mff.advanced_sampling
   :noindex:
   :members:
