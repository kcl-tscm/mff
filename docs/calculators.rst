Calculators
===========

A mapped potential is a tabulated 2- or 3-body interatomic potential created using Gaussian process regresssion and a 2- or 3-body kernel.
To use a mapped potential created with this python package within the ASE environment, it is necessary to setup a calculator using the ``mff.calculators`` class.

Theory/Introduction
-------------------

The ``model.build_grid()`` function builds a taulated 2- or 3- body potential. The calculator mthod allows to exploit the ASE functionalities to 
run molecular dynamics simulations.
For a 2-body potential, the calculator class computes the energy and force contributions to a central atom for each other atom in its neighbourhood.
These contributions depend only on the interatomic pairwise distance, and are computed trhough 1D spline interpolation of the stored values of the pairwise energy. The magnitude and verse of the pairwise force contributions are computed using the analytic derivitive of this 1D spline, while the direction of the force contribution must be the line that connects the central atom and its neighbour, for symmetry.
When a 3-body potential is used, the local energy and force acting on an atom are a sum of triplet contributions which contain the central atom and two other atoms within a cutoff distance. The triplet energy contributions are computed using a 3D spline interpolation on the stored values of triplet energy which have been calculated using  ``model.build_grid()``. The local force contributions are obtained through analytic derivative of the 3D spline interpolation used to calculate triplet energies.
The calculator behaves like a tabulated potential, and its speed scales linearly (2-body) or quadratically (3-body) with the number of atoms within a cutoff distance, and is completely independent of the number of training points used for the Gaussian process regression.
The force field obtained is also analytically energy conserving, since the force is the opposite of the analytic derivative of the local energy.
When using a 2- or 2- + 3-body force field, the ``rep_alpha`` parameter allows the user to include a Lennard-Jones like repulsive term that adds a 2-body repulsion: $$E_{rep}(r) =  0.5 ( \text{rep_alpha}/r )^{12} $$.
This introduces a repulsive term that impedes atomic collisions when the interatomic distances fall under the region where data is available. This is especially useful for high tempoerature simulations. Default ``rep_alpha`` is zero.
In the case of a multi-species force field, the ``rep_alpha`` parameter is common to every pair of elements in the current version of the code.

WARNING: The atoms in the ``atoms`` object must be ordered in increasing atomic number for the calculator to work correctly.
To do so, simply run the following line of code on the ``atoms`` object before the calculator is assigned::

>>> atoms = atoms[np.argsort(atoms.get_atomic_numbers())]


Example
-------

Assuming we already trained a model named ``model`` and built the relative mapped force field, we can assign an ASE calculator based on such force field to an ASE atoms object. 

For a 2-body single species model::

>>> from mff.calculators import TwoBodySingleSpecies
>>> calc = TwoBodySingleSpecies(r_cut, model.grid, rep_alpha = 1.5)
>>> atoms = atoms[np.argsort(atoms.get_atomic_numbers())]
>>> atoms.set_calculator(calc)

For a 3-body model::

>>> from mff.calculators import ThreeBodySingleSpecies
>>> calc = ThreeBodySingleSpecies(r_cut, model.grid)
>>> atoms = atoms[np.argsort(atoms.get_atomic_numbers())]
>>> atoms.set_calculator(calc)

For a combined (2- + 3-body) model::

>>> from mff.calculators import CombinedSingleSpecies
>>> calc = CombinedSingleSpecies(r_cut, model.grid_2b, model.grid_3b, rep_alpha = 1.5)
>>> atoms = atoms[np.argsort(atoms.get_atomic_numbers())]
>>> atoms.set_calculator(calc)


.. automodule:: mff.calculators
   :noindex:
   :members:
