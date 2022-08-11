## **Overview**

Hi Everyone!

This repository is a statistical software package written in Cython. The matrix algebra for solving the OLS equation and statistical formulas are all coded by hand, leveraging Cython's static variable declaration and memory views for more efficient operations.

My Auto Linear Transformation (ALTOLS) algorithm is also included in this package. While the traditional OLS approach assumes all independent variables have a linear relationship with the response variable, this tests for non linear relationships to see if this leads to an improvement in model performance. 

Full documentation is available in the package with the help command.

## **Installation**

* 1. Microsoft Visual C++ 14.0 or greater is required. Get it with Microsoft C++ Build Tools. Can be installed through Visual Studio Installer or go to this link: https://visualstudio.microsoft.com/visual-cpp-build-tools/

pip install MASA
