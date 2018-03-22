from numpy.distutils.core import setup, Extension

tricub = Extension('m_ff.remapping._tricub', sources=['m_ff/remapping/_tricub.pyf', 'm_ff/remapping/_tricub.c'])

setup(
    name='m_ff',
    version='0.9.0',
    description='This is a demo package',
    author=['Aldo Glielmo', 'Claudio Zeni', 'Adam Fekete'],
    packages=['m_ff'],
    # install_requires=['scipy', 'numpy', 'future'],
    ext_modules=[tricub],
    requires=['numpy']
)

