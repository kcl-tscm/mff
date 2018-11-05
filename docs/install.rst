Installation
============

To install from source, uncompress the source files and, from the directory containing `setup.py`, run the following command::
    
    python setup.py install

Or, to build in place, run::
    
    python setup.py build_ext --inplace

If you build in place, you will also need to add your eqtools folder to your PYTHONPATH shell variable::
    
    export PYTHONPATH=$PYTHONPATH:/path/to/where/you/put/


Requirements
------------

* Python
* Theano
* Numpy
* Scipy
* Pathos
* ASE
* Asap3


Usage
-----

Description on how to use the package::

 import mff

