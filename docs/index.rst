=====================
M-FF's documentation!
=====================

Short description

>>> # Basic example

CODE FUNCTIONS

1) Read XYZ file and transform it into ASE object
Input: XYZ file
Output: ASE atoms object, total number of steps
Notes: Support fot other formats rather than XYZ could be implemented in the fututre, we need to disuss the implications of this.

2) Carve configurations, forces and global energies
Input: ASE atoms object, total number of steps, total umber of configurations to extract (default = 3000?)
Output: confs, forces, energies .npy files
Notes: It would be nice to maintin the kind of sequential configurations extraction that is already implemented and also to keep the ratio of type of elements in the central atoms possibly proportional to the suqare root of their relative presence (as it is already in the code).

3) Train and test the GP
Input: Force, configurations (possibily energies?), cutoff,  number of training points, number of testing points (default = 0), hyperparameters sigma (landscape length) , lambda (noise), theta (cutoff decay) , also with a default value.
Output: Trained GP and GP error on test
Notes: Need to optimize the process itself, add vutoff functions to the theano kernels, and need to allow for multi-processing

4) Build the mapped energy grid
Input: Cutoff, minimum distance (default = 1.5), number of points in the grid (default 100?), trained GP with its hyperparameters, elements.
utput: 4 (mono-element) or 7 (bi-element) remapped grids (in a single file), parameters used to build it written somewhere for reference.
Notes: Built using r1, r2, r12, we need to see if there is a way to exploit symmetry and reduce the number of computations needed.

5) Force and energy interpolator
Input: configurations, mapped potential grids
Output: forces and/or energies
Notes: Use the configuration carving of function 2 to create inputs for the force and energy interpolator.
We need to update the code on this since we are now working on r1 r2 r12 instead of r1 r2 $\theta$.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   confs
   gp
   grid
   tutorials




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
