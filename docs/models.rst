Models
======

The models are the classes used to build, train and test a Gaussian process, and to then build the relative mapped potential.
There are six types of models at the moment, each one is used to handle 2-, 3-, or 2+3-body kernels in the case of one or two atomic species.
When creating a model, it is therefore necessary to decide a priori the type of Gaussian process and, therefore, the type of mapped potential we want to obtain.


..  _model_build:

Building a model
----------------
To create a model based on a 2-body kernel for a monoatomic system::

    from m_ff import models
    mymodel = models.TwoBodySingleSpecies(atomic_number, cutoff_radius, sigma, theta, noise)

where the parameters refer to the atomic number of the species we are training the GP on, the cutoff radius we want to use, the lengthscale hyperparameter of the Gaussian Process, the hyperparameter governing the exponential decay of the cutoff function, and the noise associated with the output training data.
In the case of a 2+3-body kernel for a monoatomic system::

    from m_ff import models
    mymodel = models.CombinedSingleSpecies(atomic_number, cutoff_radius, sigma_2b, theta_2b, sigma_3b, theta_3b, noise)

where we have two additional hyperparameters since the lengthscale value and the cutoff decay ratio of the 2- and 3-body kernels contained inside the combined Gaussian Process are be independent.

When dealing with a two-element system, the syntax is very similar, but the ``atomic_number`` is instead a list containing the atomic numbers of the two species, in increasing order::

    from m_ff import models
    mymodel = models.CombinedTwoSpecies(atomic_numbers, cutoff_radius, sigma_2b, theta_2b, sigma_3b, theta_3b, noise)


Fitting the model
-----------------
Once a model has been built, we can train it using a dataset of forces, energies, or energies and forces, that has been created using the ``configurations`` module. If we are training only on forces::

    mymodel.fit(training_confs, training_forces)

training only on energies::

    mymodel.fit_energy(training_confs, training_energies)

training on both forces and energies::

    mymodel.fit_force_and_energy(training_confs, training_forces, training_energies)

Additionaly, the argument ``nnodes`` can be passed to any fit function in order to run the process on multiple processors::

    mymodel.fit(training_confs, training_forces, nnodes = 4)



Predicting forces and energies with the GP
------------------------------------------
Once the Gaussian process has been fitted, it can be used to directly predict forces and energies on test configurations. To predict the force and the energy for a single test configuration::

    force = mymodel.predict(test_configuration)
    energy = mymodel.predict_energy(test_configuration)

the boolean variable ``return_std`` can be passed to the force and energy predict functions in order to obtain also the standard deviation associated with the prediction, default is False::

    mean_force, std_force = mymodel.predict(test_configuration, return_std = True)

.. _model_map:

Building a mapped potential
---------------------------
Once the Gaussian process has been fitted, either via force, energy or joint force/energy fit, it can be mapped onto a non-parametric 2- and/or 3-body potential using the ``build_grid`` function. The ``build_grid`` function takes as arguments the minimum grid distance (smallest distance between atoms for which the potential will be defined), the number of grid points to use while building the 2-body mapped potential, and the number of points per dimension to use while building the 3-body mapped potential. 
For a 2-body model::

    mymodel.build_grid(grid start, num_2b)

For a 3-body model::

    mymodel.build_grid(grid start, num_3b)

For a combined model::

    mymodel.build_grid(grid start, num_2b, num_3b)

Additionaly, the argument ``nnodes`` can be passed to the ``build_grid`` function for any model in order to run the process on multiple processors::

    mymodel.build_grid(grid start, num_2b, num_3b, nnodes = 4)


Saving and loading a model
--------------------------
At any stage, a model can be saved using the ``save`` function that takes a .json filename as the only input::

    mymodel.save("thismodel.json")

the save function will create a .json file containing all of the parameters and hyperparameters of the model, and the paths to the .npy and .npz files containing, respectively, the saved GPs and the saved mapped potentials, which are also created by the save funtion.

To load a previously saved model of a known type (here for example a CombinedSingleSpecies model) simply run::

    mymodel = models.CombinedSingleSpecies.from_json("thismodel.json")


Model's complete reference
--------------------------
.. automodule:: m_ff.models.twobody
   :members:



.. automodule:: m_ff.models.threebody
   :members:

.. automodule:: m_ff.models.combined
   :members:
