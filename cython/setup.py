import setuptools
from distutils.core import setup
from distutils.extension import Extension
import sys
import numpy

if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')
else:
    USE_CYTHON = False
ext = '.pyx' if USE_CYTHON else '.cpp'
extensions = [Extension("src.python.cython_utils",
                        ["src/python/cython_utils"+ext],
                        compiler_directives={'language_level': 3},
                        language='c++',
                        include_dirs=['src/cpp/', numpy.get_include()])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    ext_modules = extensions
)        