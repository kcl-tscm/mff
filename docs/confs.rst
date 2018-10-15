.. _configurations:

Using M-FF Configurations
=========================

The M-FF package uses training and testing data extracted from .xyz files. The ``m_ff.configurations`` module contains the function ``carve_confs`` which is used to save .npy files containing local atomic environments, the forces acting on the central atoms of these local atomic environments and, if present, the energy associated with the snapshot the local environment has been extracted from.
To extract local atomic environments, forces, energies and a list of all the elements contained in an ase ``atoms`` object::

>>> from ase.io import read
>>> from m_ff.configurations import carve_confs
>>> traj = read(filename, format='extxyz')
>>> elements, confs, forces, energies = carve_confs(traj, r_cut, n_data)

where r_cut specifies the cutoff radius that will be applied to extract local atomic environments containing all atomis within r_cut from the central one, and n_data specifies the total number of local atomic environments to extract.



.. .. automodule:: m_ff.configurations
..    :members:
