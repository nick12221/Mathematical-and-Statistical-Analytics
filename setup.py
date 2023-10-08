from setuptools import find_packages, setup, Extension
import numpy as np


setup(
    name="MASA",
    version="1.6",
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
    xt_modules=[
        Extension(
            "MASA.MASA",
            sources=["MASA/MASA.c"],  # Include the pre-compiled binary .c file
            include_dirs=[],
        )
    ],                   
    include_dirs=np.get_include(),
    install_requires=[
        'PyObjC;platform_system=="Darwin"',
        'PyGObject;platform_system=="Linux"',
    ]
)