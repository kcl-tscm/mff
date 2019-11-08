# Machine learning nonparametric force fields (MFF)
[![Build Status](https://travis-ci.com/kcl-tscm/mff.svg?branch=master)](https://travis-ci.com/kcl-tscm/mff)
[![Doc](https://img.shields.io/badge/docs-master-blue.svg)](https://kcl-tscm.github.io/mff/)
[![DOI](https://zenodo.org/badge/123019663.svg)](https://zenodo.org/badge/latestdoi/123019663)

An example tutorial jupyter notebook can be found in the `tutorials` folder.

![alt text](https://kcl-tscm.github.io/mff/_static/mff_logo_2.svg)
## Table of Contents

- [Background on MFFs](#background)
- [Install](#install)
- [Examples](#examples)
- [Maintainers](#maintainers)
- [References](#references)

## Background on MFF

The MFF package uses Gaussian process regression to extract non-parametric 2- and 3- body force fields from ab-initio calculations.
For a detailed description of the theory behind Gaussian process regression to predict forces and/or energies, and an explanation of the mapping technique used, please refer to [1].

For an example use of the MFF package to build 3-body force fields for Ni nanoclusters, please see [2].

## Pip Installation

To install MFF with pip, simply run the following in a Python 3.6 or 3.7 environment:

    pip install mff


## Source Installation

If the pip installation fails, try the following:
Clone from source and enter the folder:

    git clone https://github.com/kcl-tscm/mff.git
    cd mff


If you don't have it, install virtualenv:

    pip install virtualenv	   


Create a virtual environment using a python 3.6 installation	

	virtualenv --python=/usr/bin/python3.6 <path/to/new/virtualenv/>	


Activate the new virtual environment 	

	source <path/to/new/virtualenv/bin/activate>	


To install from source run the following command:	

    python setup.py install	


Or, to build in place for development, run:	

    python setup.py develop



## Examples
Refer to the two files in the Tutorial folder for working jupyter notebooks showing most of the functionalities of this package.


## Maintainers

* Claudio Zeni (claudio.zeni@kcl.ac.uk),
* Aldo Glielmo (aldo.glielmo@kcl.ac.uk),
* Ádám Fekete (adam.fekete@kcl.ac.uk).

## References

[1] A. Glielmo, C. Zeni, A. De Vita, *Efficient non-parametric n-body force fields from machine learning* (https://arxiv.org/abs/1801.04823)

[2] C .Zeni, K. Rossi, A. Glielmo, A. Fekete, N. Gaston, F. Baletto, A. De Vita *Building machine learning force fields for nanoclusters* (https://arxiv.org/abs/1802.01417)
