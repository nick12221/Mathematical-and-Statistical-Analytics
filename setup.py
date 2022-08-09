from setuptools import find_packages, setup

import numpy as np
from Cython.Build import cythonize


setup(
    name="MASA Library",
    version="1.0",
    packages=find_packages(),
    author="Nick Pettit",
    description="Statistical package focused on linear regression and statistics.",
    long_description="",
    long_description_content_type='text/markdown',
    url="https://github.com/nick12221/Mathematical-and-Statistical-Analytics",
    ext_modules=cythonize(["https://github.com/nick12221/Mathematical-and-Statistical-Analytics/tree/main/cython/MASA.pyx"]),
    include_dirs=np.get_include(),
    install_requires=[
        'numpy>=1.21.5',
        'PyObjC;platform_system=="Darwin"',
        'PyGObject;platform_system=="Linux"',
        'cython>=0.29.28'
    ]
)