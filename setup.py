from setuptools import find_packages, setup

import numpy as np
from Cython.Build import cythonize


setup(
    name="MASA",
    version="1.4",
    packages=find_packages(),
    author="Nick Pettit",
    author_email='nickpettit12321@gmail.com',
    description="Statistical package focused on linear regression and statistical testing.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    project_urls={
        'Source Code': 'https://github.com/nick12221/Mathematical-and-Statistical-Analytics',},
    license='MIT',
    url="https://github.com/nick12221/Mathematical-and-Statistical-Analytics",
    ext_modules=cythonize(["MASA/MASA.pyx"],
                            compiler_directives={'language_level' : "3"}),
    include_dirs=np.get_include(),
    install_requires=[
        'PyObjC;platform_system=="Darwin"',
        'PyGObject;platform_system=="Linux"',
    ]
)