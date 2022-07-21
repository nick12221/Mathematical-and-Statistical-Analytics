import distutils.core
import Cython.Build
from setuptools import Extension
import numpy

distutils.core.setup(
    ext_modules = Cython.Build.cythonize("sample.pyx", compiler_directives={'language_level': 3}),
                                        include_dirs=[numpy.get_include()])


             