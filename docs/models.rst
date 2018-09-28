Models
======

The models are the classes used to build, train and test a Gaussian process, and to then build the mapped potential using said Gaussian Process.
There are six types of models at the moment, each one is used to handle 2-, 3-, or 2+3-body kernels in the case of one and two atomic species.
When creating a model, it is therefore necessary to decide a priori the type of Gaussian process and, subsequently, the type of mapped potential we want to obtain.

Building a model
----------------
For example, to create a model based on 2-body kernels for a monoatomic system:

>>> from m_ff import models

>>> mymodel = models.TwoBodySingleSpecies(atomic_number, cutoff_radius, sigma, theta, noise)

where the parameters refer to the atomic number of the species we are training the GP on, the cutoff radius we want to use, the lengthscale hyperparameter of the Gaussian Process, the hyperparameter governing the exponential decay of the cutoff function, and the noise associated with the output training data.
In the case of a 2+3-body kernel for a monoatomic system:

>>> from m_ff import models

>>> mymodel = models.CombinedSingleSpecies(atomic_number, cutoff_radius, sigma_2b, theta_2b, sigma_3b, theta_3b, noise)

where we have two additional hyperparameters since the lengthscale value and the cutoff decay ratio of the 2- and 3-body kernels contained inside the combined Gaussian Process can be independent.

When dealing with a two-element system, the syntax is very similar, but the ``atomic_number`` becomes a list containing the atomic numbers of the two species, in increasing order:

>>> from m_ff import models

>>> mymodel = models.CombinedTwoSpecies(atomic_numbers, cutoff_radius, sigma_2b, theta_2b, sigma_3b, theta_3b, noise)


Fitting the model
-----------------
Once the model has been built, we can train it using a dataset of forces, energies, or energies and forces, that has been created using the configurations module. If we are training only on forces:

>>> mymodel.fit(training_confs, training_forces)

training only on energies:

>>> mymodel.fit_energy(training_confs, training_energies)

training on both forces and energies:

>>> mymodel.fit_force_and_energy(training_confs, training_forces, training_energies)

Predicting forces and energies with the GP
------------------------------------------
Once the Gaussian process has been fitted, it can be used to directly predict forces and energies on test configurations. To predict the force and the energy for a test configuration:

>>> force = mymodel.predict(test_configuration)
>>> energy = mymodel.predict_energy(test_configuration)

the boolean variable ``return_std`` can be passed to the predict functions in order to obtain also the standard deviation associated with the prediction:

>>> mean_force, std_force = mymodel.predict(test_configuration, return_std = True)



.. automodule:: m_ff.models
   :members:
