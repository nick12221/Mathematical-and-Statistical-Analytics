## **Overview**

This repository is a statistical software package written in Cython focused on statistical testing and Linear Regression. The linear algebra for solving the OLS equation and statistical formulas are all coded from scratch.

My Auto Linear Transformation (ALTOLS) algorithm, a variant on the traditional OLS, is also included in this package. While the traditional OLS approach assumes all continuous independent variables have a linear relationship with the response variable, this automatically tests for non linear relationships to see if this leads to an improvement in model performance (Adjusted R2). 

All functions only accept numpy array type objects. Full documentation is available in the package with the help command. A sample jupyter notebook and dataset are included in the "Sample Notebook and Data" folder on GitHub.

## **Installation**

* 1. Just pip install and go from there!   **pip install MASA**
