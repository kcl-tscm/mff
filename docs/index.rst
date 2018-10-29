.. m-ff documentation master file, created by
   sphinx-quickstart on Thu Oct 18 11:47:17 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

M-FF's documentation
====================


M-FF is a package built to apply machine learning to atomistic simulation within an ASE environment.
M-FF uses Gaussian process regression to build non-parametric 2- and 3- body force fields from a small dataset of ab-initio simulations. These Gaussian processes are then mapped onto a non-parametric tabulated 2- or 3-body force field that can be used within the ASE environment to run atomistic simulation with the computational speed of a tabulated potential and the chemical accuracy offered by machine learning on ab-initio data.
Trajectories or snapshots of the system of interest are used to train the potential, these must contain atomic positions, atomic numbers and forces (and/or total energies), preferrabily calculated via ab-initio methods.

At the moment the package supports single- and two-element atomic environments; we aim to support three-element atomic environments in future versions.

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   install
   models
   confs
   gp
   calculators
   api

.. toctree::
   :caption: Appendix

   genindex

Indices and tables
==================

* :ref:`Index <genindex>`
* :ref:`modindex`

.. c * :ref:`search`
