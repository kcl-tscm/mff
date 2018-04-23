from numpy.distutils.core import setup, Extension


tricube_cpp_module = Extension(
    'm_ff.interpolation.tricube_cpp._tricube',
    sources=["m_ff/interpolation/tricube_cpp/tricube_module.c", "m_ff/interpolation/tricube_cpp/_tricube.c"]
)

tricube_fortran_module = Extension(
    'm_ff.interpolation.tricube_fortran._tricube',
    sources=['m_ff/interpolation/tricube_fortran/_tricube.pyf', 'm_ff/interpolation/tricube_fortran/_tricube.c']
)

setup(
    name='m_ff',
    version='0.9.0',
    description='This is a demo package',
    author=['Aldo Glielmo', 'Claudio Zeni', 'Adam Fekete'],
    packages=['m_ff'],
    # install_requires=['scipy', 'numpy', 'future'],
    ext_modules=[tricube_cpp_module, tricube_fortran_module],
    # ext_modules=[tricube_cpp_module],
    requires=['numpy']
)



