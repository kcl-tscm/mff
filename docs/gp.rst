Gaussian Processes
==================

Gaussian process regression module suited to learn and 
predict energies and forces

Example::

 gp = GaussianProcess(kernel, noise)
 gp.fit(train_configurations, train_forces)
 gp.predict(test_configurations)


.. automodule:: mff.gp
   :members:


Two Body Kernel
---------------

Module that contains the expressions for the 2-body single-species and
multi-species kernel. The module uses the Theano package to create the 
energy-energy, force-energy and force-force kernels through automatic
differentiation of the energy-energy kernel.
The module is used to calculate the energy-energy, energy-force and 
force-force gram matrices for the Gaussian processes, and supports 
multi processing.
The module is called by the gp.py script.

Example::

 from twobodykernel import TwoBodySingleSpeciesKernel
 kernel = kernels.TwoBodySingleSpeciesKernel(theta=[sigma, theta, r_cut])
 ee_gram_matrix = kernel.calc_gram_e(training_configurations, number_nodes)


.. automodule:: mff.kernels.twobodykernel
   :members:



Three Body Kernel
-----------------

Module that contains the expressions for the 3-body single-species and
multi-species kernel. The module uses the Theano package to create the 
energy-energy, force-energy and force-force kernels through automatic
differentiation of the energy-energy kernel.
The module is used to calculate the energy-energy, energy-force and 
force-force gram matrices for the Gaussian processes, and supports 
multi processing.
The module is called by the gp.py script.

Example::

 from threebodykernel import ThreeBodySingleSpeciesKernel
 kernel = kernels.ThreeBodySingleSpeciesKernel(theta=[sigma, theta, r_cut])
 ee_gram_matrix = kernel.calc_gram_e(training_configurations, number_nodes)


.. automodule:: mff.kernels.threebodykernel
   :members:
