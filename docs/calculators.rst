Calculators
===========

A mapped potential is a tabulated 2- or 3-body interatomic potential created using Gaussian process regresssion and a 2- or 3-body kernel.
To use a mapped potential created with this python package within the ASE environment, it is necessary to setup a calculator using the ``mff.calculators`` class.

Theory/Introduction
-------------------

The ``model.build_grid()" function builds a taulated 2- or 3- body potential. The calculator mthod allows to exploit the ASE functionalities to 
run molecular dynamics simulations.
For a 2-body potential, the calculator class computes the energy and force contributions to a central atom for each other atom in its neighbourhood.
These contributions depend only on the interatomic pairwise distance, and are computed trhough 1D spline interpolation of the stored values of the pairwise energy. The magnitude and verse of the pairwise force contributions are computed using the analytic derivitive of this 1D spline, while the direction of the force contribution must be the line that connects the central atom and its neighbour, for symmetry.
When a 3-body potential is used, the local energy and force acting on an atom are a sum of triplet contributions which contain the central atom and two other atoms within a cutoff distance. The triplet energy contributions are computed using a 3D spline interpolation on the stored values of triplet energy which have been calculated using  ``model.build_grid()". The local force contributions are obtained through analytic derivative of the 3D spline interpolation used to calculate triplet energies.
The calculator behaves like a tabulated potential, and its speed scales linearly (2-body) or quadratically (3-body) with the number of atoms within a cutoff distance, and is completely independent of the number of training points used for the Gaussian process regression.
The force field obtained is also analytically energy conserving, since the force is the opposite of the analytic derivative of the local energy.


Example
-------

Assuming we already trained a model and built the relative mapped force field, we can assign an ASE calculator based on such force field to an ASE atoms object.

>>> 
>>> 
>>> 


Running the Calculator
----------------------

...


.. automodule:: mff.calculators
   :noindex:
   :members:
