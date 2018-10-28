from setuptools import find_packages
from numpy.distutils.core import setup, Extension


tricube_cpp_module = Extension(
    'mff.interpolation.tricube_cpp._tricube',
    sources=["mff/interpolation/tricube_cpp/tricube_module.c", "mff/interpolation/tricube_cpp/_tricube.c"],
)

tricube_fortran_module = Extension(
    'mff.interpolation.tricube_fortran._tricube',
    sources=['mff/interpolation/tricube_fortran/_tricube.pyf', 'mff/interpolation/tricube_fortran/_tricube.c'],
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='m-ff',
    version='0.9.0',
    author= 'Claudio Zeni, Adam Fekete, Aldo Glielmo',
    description='This is a demo package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kcl-tscm/M-FF",
    packages=find_packages(),
    ext_modules=[tricube_cpp_module, tricube_fortran_module],
    requires=['numpy', 'asap3', 'ase', 'theano', 'scipy']
    # install_requires=['scipy', 'numpy', 'future'],
)


# import setuptools
#
# with open("README.md", "r") as fh:
#     long_description = fh.read()
#
# setuptools.setup(
#     name="example_pkg",
#     version="0.0.1",
#     author="Example Author",
#     author_email="author@example.com",
#     description="A small example package",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     url="https://github.com/pypa/example-project",
#     packages=setuptools.find_packages(),
#     classifiers=(
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: MIT License",
#         "Operating System :: OS Independent",
#     ),
# )
