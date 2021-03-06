'''
The MathematicalAndStatisticalAnalytics package provides:
1. Functions for performing linear algebra relevant to OLS regression.
2. Class for OLS regression modelling and new variation called Auto Linear Fit.
3. Functions for OLS model performance metrics.
4. Functions for statistics and statistical testing.


NumPy is needed for this package.The array object structure is . For dataframes this can 
be done using .values, and for lists numpy.array().

What is included in the documentation:

Docstrings for all functions are included. The docstrings can be accessed
using the help command.
'''


###-------------------------------------Import Needed Packages------------------------------------###
import numpy as np
cimport numpy as np
cimport cython


###----------------------------------------Fused Datatypes----------------------------------------###
ctypedef fused fusedDType:
    int
    double
    long long


ctypedef fused fusedDType2:
    int
    double
    long long

###-------------------------------Dictionary of Linear Transformations----------------------------###


LinearTransformationDict = {'square': np.square,
                            'squareRoot': np.sqrt,
                            'reciprocal': np.reciprocal,
                            'naturalLog': np.log}


###-------------------------------------OLS Lin Alg Functions-------------------------------------###


def GetMatrix(long long int rows, long long int columns):
    '''This function creates a numpy ndarray of n X m specified dimensions. A random number generator
    is used to generate datapoints.

    Parameters:

    rows: A value for the number of rows in the generated matrix. 

    columns: A value for the number of columns in the generated matrix.

    Returns: A numpy ndarray of specified dimensions.
    '''

    result = np.zeros((rows, columns))

    cdef Py_ssize_t i, j

    for i in range(rows):
        for j in range(columns):
            x = np.random.rand()  
            result[i, j] = x

    return result


def floatConverter(fusedDType[:,:] intMatrix):
    cdef double[:,:] floatMatrix = np.array(intMatrix, dtype=np.double)
    return floatMatrix


def floatConverter1DArray(fusedDType[:] intMatrix):
    cdef double[:] floatMatrix = np.array(intMatrix, dtype=np.double)
    return floatMatrix


def GetIdentityMatrix(long long int dimensions):
    '''This function creates an identity matrix of n X n dimensions. An identity matrix has 1 across 
    its diagonal and zeroes in all other locations. 

    Parameters:

    dimensions: An integer value for the dimensions of the n X n square matrix. 

    Returns: A numpy ndarray identity matrix of n X n dimensions.
    '''
    
    identityMatrix = np.zeros((dimensions, dimensions))

    cdef Py_ssize_t i

    for i in range(dimensions):
        identityMatrix[i, i] = 1
            
    return identityMatrix


def GetTransposeMatrix(fusedDType[:, :] matrix1):
    '''This function transposes the rows into columns for a given matrix. The data in the 
    matrix can be of int or float type.

    Parameters: 

    matrix1: A n X m matrix or vector. 

    Returns: A transposed version of the original matrix or vector passed through the function.
    '''

    newRows = matrix1.shape[1]
    newCols = matrix1.shape[0]

    cdef double[:, :] matrix
    matrix = np.zeros((newRows, newCols), dtype=np.double)

    matrix = floatConverter(matrix1)

    cdef Py_ssize_t i, j

    transposedMatrix = np.zeros((newRows,newCols), dtype=np.double) 
    cdef double[:, :] resultView = transposedMatrix

    for i in range(newRows):
        for j in range(newCols):
            resultView[i, j] = matrix[j, i]
    
    return transposedMatrix


def GetDotProduct(fusedDType[:,:] matrix1, fusedDType2[:,:] matrix2):
    '''This function calculates the dot product of two matrices. The data in both 
    matrices can be of int or float type.s
    
    Parameters:
    
    matrix1: This is a matrix of n x m dimensions. These can be any dimensions, as long as the
            number of columns in this matrix equals the number of rows in the second matrix.
    
    matrix2: This is a matrix of m x p dimensions. These can be any dimensions, as long as the
            number of rows in this matrix equals the number of columns in the first matrix.
    
    Returns: A square matrix of m X m dimensions that is the result of the dot product calculated
            from the first and second matrix.
    '''
    assert matrix1.shape[1] == matrix2.shape[0], "The number of columns in matrix 1 must match the rows in matrix 2."

    firstMatrixRows = matrix1.shape[0]
    secondMatrixCols = matrix2.shape[1]
    secondMatrixRows = matrix2.shape[0]

    cdef double[:, :] matrix1Float
    matrix1Float = np.zeros((firstMatrixRows, matrix1.shape[1]), dtype=np.double)

    cdef double[:, :] matrix2Float
    matrix2Float = np.zeros((secondMatrixRows, secondMatrixCols), dtype=np.double)

    matrix1Float = floatConverter(matrix1)
    matrix2Float = floatConverter(matrix2)

    cdef Py_ssize_t i, j, k
    cdef double number

    dotProdMatrix = np.zeros((firstMatrixRows,secondMatrixCols),dtype=np.double) 
    cdef double[:, :] resultView = dotProdMatrix

    for i in range(firstMatrixRows):
        for j in range(secondMatrixCols):
            number = 0
            for k in range(secondMatrixRows):
                number += matrix1[i, k] * matrix2[k, j]

            resultView[i, j] = number
        
    return dotProdMatrix


def GetMatrixDiagonal(fusedDType[:,:] matrix1):
    '''This function extracts the diagonal of a square matrix that can have any n X n dimensions.
    The data of the matrix can be of int or float type.

    Parameters:

    matrix1: An n X n matrix that can be of any size. 

    Returns: A vector that is the diagonal of the matrix passed through the function.
    '''

    assert matrix1.shape[0] == matrix1.shape[1]

    MatrixCols = matrix1.shape[1]

    cdef double[:, :] matrix
    matrix = np.zeros((MatrixCols, MatrixCols), dtype=np.double)

    cdef Py_ssize_t i
    cdef Py_ssize_t z = 0

    matrix = floatConverter(matrix1)

    diagonalMatrix = np.zeros((1,MatrixCols),dtype=np.double) 
    cdef double[:, :] resultView = diagonalMatrix

    for i in range(MatrixCols):
        resultView[z, i] = matrix[i, i] 

    return diagonalMatrix


def GetDeterminant(fusedDType[:,:] matrix1):
    '''This function calculates the determinant of a square matrix. The data of 
    the matrix can be of int or float type.

    Parameters:

    matrix1: A square matrix.

    Returns: The determinant of the matrix passed into the function.
    '''
    
    matrixLength = matrix1.shape[0]

    if fusedDType is int:
        dtype = np.intc
    elif fusedDType is double:
        dtype = np.double
    elif fusedDType is cython.longlong:
        dtype = np.longlong

    cdef fusedDType[:, :] matrix
    matrix = np.zeros((matrixLength, matrixLength), dtype=dtype)

    matrix[:,:] = matrix1

    cdef double[:, :] matrix2
    matrix2 = np.zeros((matrixLength, matrixLength), dtype=np.double)

    cdef double[:, :] matrix3
    matrix3 = np.zeros((matrixLength, matrixLength), dtype=np.double)

    cdef double[:] oldFirstRow
    oldFirstRow = np.zeros((matrixLength), dtype=np.double)

    cdef double[:] newFirstRow
    newFirstRow = np.zeros((matrixLength), dtype=np.double)
    
    matrix2 = floatConverter(matrix)
    matrix3[:,:] = matrix2

    cdef Py_ssize_t i, j, k, x, newRow
    cdef double rowChange
    cdef double rowScaler
    cdef double changeIndicator
    cdef double determinant

    if sum([sub[0] for sub in matrix2]) == 0:
        return 0
    elif matrixLength == 1:
        return matrix2[0, 0]
    else:
        rowChange = 0
        if matrix2[0, 0] == 0:
            FirstColumnList = matrix3[:,0]
            newRow = next(x for x, val in enumerate(FirstColumnList) if val > 0)
            oldFirstRow = matrix3[0,:]
            newFirstRow = matrix3[newRow,:]
            for j in range(matrixLength):
                matrix2[0, j] = newFirstRow[j]
                matrix2[newRow, j] = oldFirstRow[j]
            rowChange = 1
        for i in range(matrixLength):
            for j in range(i+1, matrixLength):
                if matrix2[i, i] == 0:
                    return 0
                RowScaler = matrix2[j, i] / matrix2[i, i]
                for k in range(matrixLength): 
                    matrix2[j, k] = matrix2[j, k] - matrix2[i, k] * RowScaler
    
    changeIndicator = -1           
    determinant = (changeIndicator)**rowChange*np.prod(GetMatrixDiagonal(matrix2))

    return determinant


def GetMatrixMinor(double[:,:] matrix1, Py_ssize_t i, Py_ssize_t j):
    '''This function calculates the minor of a specified matrix. A matrix minor excludes a 
    specific row and column from the original matrix. The data of the matrix can be of 
    int or float type.

    Parameters:

    matrix1: Matrix to pass through the function.

    i: index of row of matrix to exclude for the minor.

    j: index of column of matrix to exclude for the minor.

    Returns: The matrix minor of the input matrix.
    '''

    matrixMinor = np.delete(matrix1, i, 0)
    matrixMinor = np.delete(matrixMinor, j, 1)

    return matrixMinor


def GetMatrixInverse(fusedDType[:, :] matrix1):
    '''This function calculates the inverse of a matrix. 

    Parameters:

    matrix1: An n X n square matrix.

    Returns: The inverse of the matrix passed through the function.
    '''

    assert matrix1.shape[1] == matrix1.shape[0], "Dimensions must be square."

    matrixLength = matrix1.shape[0]

    cofactorMatrix = np.zeros((matrixLength, matrixLength), dtype=np.double) 
    cdef double[:, :] cofactors = cofactorMatrix

    matrix2 = np.zeros((matrixLength, matrixLength), dtype=np.double)
    cdef double[:, :] resultView = matrix2

    matrix2[:,:] = matrix1

    matrix2 = floatConverter(matrix1)

    cdef double determinant
    determinant = GetDeterminant(matrix2)

    cdef Py_ssize_t row, col
    cdef double negNumber, posNumber
    negNumber = -1
    posNumber = 1
    
    if matrixLength == 1:
        return np.array([[posNumber/matrix2[0, 0]]])
    
    if matrixLength == 2:
        return np.array([[matrix2[1, 1]/determinant, negNumber*matrix2[0, 1]/determinant],
                        [negNumber*matrix2[1, 0]/determinant, matrix2[0, 0]/determinant]])

    for row in range(matrixLength):
        for col in range(matrixLength):
            minor = GetMatrixMinor(resultView,row,col)
            cofactors[row, col] = ((negNumber)**(row+col)) * GetDeterminant(minor)
    
    cofactorMatrix = GetTransposeMatrix(cofactorMatrix)

    for row in range(matrixLength):
        for col in range(matrixLength):
            cofactorMatrix[row, col] = cofactorMatrix[row, col]/determinant
        
    return cofactorMatrix


###----------------------------------------OLS Model Class----------------------------------------###


class OLSSuite:
    '''This class is designed for building OLS model objects. OLSSuite is built on top of numpy arrays,
    leveraging the ndarray object for matrix operations. Cython is also used for more efficient matrix 
    operations. There are three primary methods and six attributes in this class.

    Methods:

    GetOLSResults: This function fits the model object with the OLS results of a specified independent 
                   and dependent variables. All six attributes are populated, with the primary results  
                   being stored in the OLSResults dictionary. Two dimensional arrays must be passed 
                   for the x and y variables.

    GetAltOLS: This method is a variation of the standard OLS methodology. Instead of assuming all 
               independent variables have a linear relationship with the dependent variable, this 
               algorithm tests different relationships. If an improvement in R2 greater than the 
               benchmark rate, then that transformation will automatically be applied and OLS model 
               results from the auto fitted model will be returned.

    GetPredictions: Method for making predictions based on the fitted model object. Model object must 
                    first run GetOLSResults or GetAltOLS before making predictions.

    Attributes: All attributes are returned as numerical values or numpy ndarrays.

    OLSResult: A dictionary of the regression results.

    Coefficients: List of the values for the coefficients in the order variables were 
                  past into the model.

    stdErrorCoefs: The standard errors for the model coeffiients.

    R2Score: R2 score of the model.
    
    AdjR2Score: Adjusted R2 score of the model.

    dofResiduals: The degree of freedom for the residuals.
    '''

    def __init__(self):
        '''This function initializes six attributes for the OLSSuite class.

        Attributes:
        
        OLSResult: A dictionary of the regression results.

        Coefficients: List of the values for the coefficients in the order variables were 
                      past into the model.

        stdErrorCoefs: The standard errors for the model coeffiients.

        R2Score: R2 score of the model.
        
        AdjR2Score: Adjusted R2 score of the model.

        dofResiduals: The degree of freedom for the residuals.
        '''

        self.OLSResults = None
        self.Coefficients = None
        self.stdErrorCoefs = None

        self.Metrics = None
        self.mse = None
        self.R2Score = None
        self.AdjR2Score = None
        self.dofResiduals = None
        self.nObservations = None
        self.MeanAbsError = None
        


    def GetCoefs(self, fusedDType[:, :] y, fusedDType2[:, :] x):
        '''This function calculates the beta coefficients for the OLS Model. 
    
        Parameters:
    
        y: Response variable to pass through the model. Can pass a numpy array or list object. Dataframes 
           must be converted to an array or list before being passed through the function.

        x: Independent variables to pass through the model. Can pass a numpy array or list object. Dataframes 
           must be converted to an array or list before being passed through the function.

        Returns: Coefficients attribute for the model object.
        '''
        
        cdef double[:, :] xMatrix
        xMatrix = np.zeros((x.shape[0], x.shape[1]), dtype=np.double)

        cdef double[:, :] yMatrix
        yMatrix = np.zeros((y.shape[0], y.shape[1]), dtype=np.double)

        yMatrix = floatConverter(y)
        xMatrix = floatConverter(x)

        xTransposed = GetTransposeMatrix(xMatrix)
        self.Coefficients = GetDotProduct(GetDotProduct(GetMatrixInverse((GetDotProduct(xTransposed, xMatrix))),xTransposed),yMatrix)

        
    def GetPredictions(self, fusedDType[:, :] x):
        '''This function calculates the predictions for a given dataset. 
    
        Parameters:

        x: Independent variables to pass through the model. Must pass a numpy array. 
           Dataframes must be converted to an array before being passed through the function.
    
        Returns: Numpy array of predictions from the fitted model object.
        '''

        cdef double[:, :] xMatrix
        xMatrix = np.zeros((x.shape[0], x.shape[1]), dtype=np.double)

        xMatrix = floatConverter(x)
    
        preds = GetDotProduct(GetTransposeMatrix(self.Coefficients), GetTransposeMatrix(xMatrix))
        predictions = np.reshape(preds, (x.shape[0], 1))
    
        return predictions


    def GetCoefStdErrors(self, fusedDType[:, :] y, fusedDType2[:, :] x):
        '''This function calculates the standard errors for the beta coefficients. 
    
        Parameters:
    
        y: Response variable to pass through the model. Can pass a numpy array or list object. Dataframes 
           must be converted to an array or list before being passed through the function.

        x: Independent variables to pass through the model. Can pass a numpy array or list object. Dataframes 
           must be converted to an array or list before being passed through the function.
    
        Returns: Coefficient Standard Errors attribute for the model object.
        '''

        cdef double[:, :] xMatrix
        xMatrix = np.zeros((x.shape[0], x.shape[1]), dtype=np.double)

        cdef double[:, :] yMatrix
        yMatrix = np.zeros((y.shape[0], y.shape[1]), dtype=np.double)

        yMatrix = floatConverter(y)
        xMatrix = floatConverter(x)

        cdef double mse
        
        invertedMatrix = GetMatrixInverse(GetDotProduct(GetTransposeMatrix(xMatrix), xMatrix))
        diagonalMatrix = GetMatrixDiagonal(invertedMatrix)
        self.mse = GetMSE(yMatrix, self.GetPredictions(xMatrix))
        self.stdErrorCoefs = (self.mse*diagonalMatrix)**.5


    def GetOLSResults(self, fusedDType[:, :] y, fusedDType2[:, :] x, xVariables=[]):
        '''This function calculates the OLS equation for the given dependent and independent variables.
         A summary of the model is printed and a dictionary is created to store all OLS results. Categorical 
        variables must be dummied before passing data through the model. Must add constant before running model.
        
        Parameters:
    
        y: Response variable to pass through the model. Can pass a numpy array or list object. Dataframes 
           must be converted to an array or list before being passed through the function.

        x: Independent variables to pass through the model. Can pass a numpy array or list object. Dataframes 
           must be converted to an array or list before being passed through the function.

        Optional Parameters:
    
        xVariables: List of the independent variable names in the order they were passed into the model.
    
        Returns: A dictionary with coefficients, standard errors, t value, and variable names.
        '''

        cdef Py_ssize_t i

        cdef double[:, :] xMatrix
        xMatrix = np.zeros((x.shape[0], x.shape[1]), dtype=np.double)

        cdef double[:, :] yMatrix
        yMatrix = np.zeros((y.shape[0], y.shape[1]), dtype=np.double)

        tValues = np.zeros((x.shape[1]), dtype=np.double)
        cdef double[:] tView = tValues      

        yMatrix = floatConverter(y)
        xMatrix = floatConverter(x)
    
        self.GetCoefs(yMatrix, xMatrix)
        self.R2Score = GetR2(yMatrix, self.GetPredictions(xMatrix))
        self.AdjR2Score = GetAdjustedR2(self.R2Score, x.shape[0], x.shape[1])
        self.GetCoefStdErrors(yMatrix, xMatrix)
        self.dofResiduals = x.shape[0] - x.shape[1]
        self.MeanAbsError = GetMeanAbsError(yMatrix, self.GetPredictions(xMatrix))
        self.nObservations = x.shape[0]

        coefficients = np.ndarray.flatten(self.Coefficients)
        stdErrors = np.ndarray.flatten(self.stdErrorCoefs)

        for i in range(coefficients.shape[0]):
            tValues[i] = coefficients[i] / stdErrors[i]
    
        if len(xVariables) > 0:
            xVariables = xVariables
        else:
            xVariables = ['x_' + str(i) for i in range(coefficients.shape[0])]
    
        self.OLSResults = {'Variables': xVariables,
                   'Model Coefficients': coefficients,
                   'Coef Std. Errors': stdErrors,
                   'T Stat':tValues
                   }

        self.Metrics = {'No. Observations':self.nObservations,
                        'DoF Residuals':self.dofResiduals,
                        'Mean Squared Error':self.mse,
                        'R2':self.R2Score,
                        'AdjR2Score':self.AdjR2Score,
                        'Mean Abs. Error':self.MeanAbsError
                        }


###----------------------------------Regression Metrics Functions---------------------------------###


def GetMSE(fusedDType[:, :] yTrue, fusedDType2[:, :] yPred):
    '''This function calculates the Mean Squared Error for a given Ordinary Least Squares model. 

    Parameters:

    yTrue: Actual y values that were used to train the OLS model. yTrue can be passed as a list,
            array, or a pandas series. Must be integer or float values.

    yPred: Predicted y values that were calculated using the trained OLS model. yPred can be
            passed as a list, array, or a pandas series. Must be integer or float values.

    Returns: The MSE of a given model.
    '''

    cdef double MSE
    cdef Py_ssize_t i, j
    j = 0

    squaredError = np.zeros((yTrue.shape[0], yTrue.shape[1]), dtype=np.double) 
    cdef double[:, :] resultView = squaredError

    cdef double[:, :] yTrueMatrix
    yTrueMatrix = np.zeros((yTrue.shape[0], yTrue.shape[1]), dtype=np.double)

    cdef double[:, :] yPredMatrix
    yPredMatrix = np.zeros((yTrue.shape[0], yTrue.shape[1]), dtype=np.double)

    yTrueMatrix = floatConverter(yTrue)
    yPredMatrix = floatConverter(yPred)

    for i in range(yTrueMatrix.shape[0]):
        resultView[i, j] = (yTrueMatrix[i, j] - yPredMatrix[i, j])**2

    MSE = np.sum(squaredError)/yTrueMatrix.shape[0]

    return MSE


def GetR2(fusedDType[:, :] yTrue, fusedDType2[:, :] yPred):
    '''This function calculates the R2 for a given Ordinary Least Squares model. R2 is the amount of variance
    explained by the linear regression model. It can be negative and cannot take a value higher than 1. 

    Parameters:

    yTrue: Actual y values that were used to train the OLS model. yTrue can be passed as a list,
           array, or a pandas series. Must be integer or float values.

    yPred: Predicted y values that were calculated using the trained OLS model. yPred can be
           passed as a list, array, or a pandas series. Must be integer or float values.

    Returns: R2 value of a given model.
    '''

    cdef double r2, posOne
    cdef Py_ssize_t i, j
    j = 0
    posOne = 1

    squaredError = np.zeros((yTrue.shape[0], yTrue.shape[1]), dtype=np.double) 
    cdef double[:, :] resultView = squaredError

    totalVariability = np.zeros((yTrue.shape[0], yTrue.shape[1]), dtype=np.double) 
    cdef double[:, :] resultView2 = totalVariability

    cdef double[:, :] yTrueMatrix
    yTrueMatrix = np.zeros((yTrue.shape[0], yTrue.shape[1]), dtype=np.double)

    cdef double[:, :] yPredMatrix
    yPredMatrix = np.zeros((yTrue.shape[0], yTrue.shape[1]), dtype=np.double)

    yTrueMatrix = floatConverter(yTrue)
    yPredMatrix = floatConverter(yPred)

    for i in range(yTrue.shape[0]):
        resultView[i, j] = (yTrueMatrix[i, j] - yPredMatrix[i, j])**2

    yMean = np.mean(yTrue)
    
    for i in range(yTrue.shape[0]):
        resultView2[i, j] = (yTrueMatrix[i, j] - yMean)**2

    if np.sum(totalVariability) == 0:
        return np.nan
    else:
        r2 = posOne - np.sum(squaredError)/np.sum(totalVariability)

    return r2


def GetAdjustedR2(double r2, double n, double p):
    '''This function calculates the Adjusted R2 of a regression output. The p parameter 
    for number of predictors should include the intercept in the count.

    Parameters:

    r2: The R2 score from the model.

    n: Sample size in the corresponding OLS Model. Must be an integer.

    p: Number of predictors used in the OLS model including the intercept. Must be an integer.

    Returns: The Adjusted R2 score.
    '''

    cdef double posOne
    posOne = 1

    adjR2 = posOne - ((posOne - r2)*(n-posOne))/(n-p)

    return adjR2


def GetMeanAbsError(fusedDType[:, :] yTrue, fusedDType2[:, :] yPred):
    '''This function calculates the Mean Absolute Error for a given Ordinary Least Squares model. 

    Parameters:

    yTrue: Actual y values that were used to train the OLS model. yTrue can be passed as a list,
            array, or a pandas series. Must be integer or float values.

    yPred: Predicted y values that were calculated using the trained OLS model. yPred can be
            passed as a list, array, or a pandas series. Must be integer or float values.

    Returns: The MAE of a given model.
    '''

    cdef double negOne
    cdef Py_ssize_t i, j
    j = 0

    cdef double[:, :] yTrueMatrix
    yTrueMatrix = np.zeros((yTrue.shape[0], yTrue.shape[1]), dtype=np.double)

    cdef double[:, :] yPredMatrix
    yPredMatrix = np.zeros((yTrue.shape[0], yTrue.shape[1]), dtype=np.double)

    yTrueMatrix = floatConverter(yTrue)
    yPredMatrix = floatConverter(yPred)

    absoluteError = np.zeros((yTrue.shape[0], yTrue.shape[1]), dtype=np.double) 
    cdef double[:, :] resultView = absoluteError   

    for i in range(yTrue.shape[0]):
        resultView[i, j] = (yTrueMatrix[i, j] - yPredMatrix[i, j])

    absoluteError = np.absolute(absoluteError)
    meanAbsoluteError = np.mean(absoluteError)

    return meanAbsoluteError


def GetVif(fusedDType[:, :] x):
    '''This function calculates the vif scores for all independent variables, which measures multicollinearity.

    Parameters:

    x: A matrix of independent variables that are used in a specified OLS model. Each variable will be used
        as the response variable with the others used as predictors to calculate the vif scores.

    Returns: A list of vif scores for each independent variable. Constant does not get a vif score.
    '''

    cdef double[:, :] xMatrix
    xMatrix = np.zeros((x.shape[0], x.shape[1]), dtype=np.double)

    xMatrix = floatConverter(x)

    vifScores = np.zeros((xMatrix.shape[1]), dtype=np.double) 
    cdef double[:] resultView = vifScores

    cdef Py_ssize_t i
    cdef double posOne
    posOne = 1

    modelVif = OLSSuite()

    for i in range(x.shape[1]):
        yNew =  xMatrix[:, i]
        yNew = np.reshape(yNew, (xMatrix.shape[0], 1))
        xNew = np.delete(xMatrix, i, 1)
        modelVif.GetOLSResults(yNew, xNew)
        resultView[i] = posOne/(posOne-modelVif.R2Score)

    return vifScores


###---------------------------------Statistical Testing Functions---------------------------------###


def GetVariance(fusedDType[:] numbers):
    '''This function calculates the variance of a list or array of numbers.

    Parameters:

    numbers: The numbers you want to calculate the variance for. Can pass a list, array, or pandas series.

    Returns: Variance of the numbers that were passed through the function.
    '''

    cdef double total, mean, negOne, variance
    cdef Py_ssize_t i

    cdef double[:] numbersMatrix
    numbersMatrix = np.zeros((numbers.shape[0]), dtype=np.double)

    numbersMatrix = floatConverter1DArray(numbers)

    negOne = -1
    total = 0
    mean = np.mean(numbersMatrix)

    for i in range(numbersMatrix.shape[0]):
        total += (numbersMatrix[i] - mean)**2

    variance = total/(numbersMatrix.shape[0]+negOne)

    return variance


def GetPearsonCorr(fusedDType[:] x, fusedDType2[:] y):
    '''This function calculates the pearson correlation coefficient between two variables.

    Parameters:

    x: First variable to pass to the function. Can be a list, array, or pandas series.

    y: Second variable to pass to the function. Can be a list, array, or pandas series.

    Returns: One number indicating the correlation coefficient between two variables.
    '''

    cdef double xMean, yMean, numerator, xSpread, ySpread, r
    cdef Py_ssize_t i 

    cdef double[:] xMatrix
    xMatrix = np.zeros((x.shape[0]), dtype=np.double)

    cdef double[:] yMatrix
    yMatrix = np.zeros((y.shape[0]), dtype=np.double)

    yMatrix = floatConverter1DArray(y)
    xMatrix = floatConverter1DArray(x)
    
    xMean = np.mean(xMatrix)
    yMean = np.mean(yMatrix)

    numerator = 0
    xSpread = 0
    ySpread = 0

    for i in range(x.shape[0]):
        numerator += (xMatrix[i] - xMean)*(yMatrix[i] - yMean)
        xSpread += (xMatrix[i] - xMean)**2
        ySpread += (yMatrix[i] - yMean)**2

    r = numerator/(xSpread*ySpread)**.5

    return r


def Get1SampleTTest(fusedDType[:] x, double popMean):
    '''This function calculates a one sample t test for testing if the mean of a sample is 
    stastistically different than a given population mean.

    Parameters:

    x: sample of data to test if mean is different than hypothesized value. Must convert    
        objects to a list or numpy array before passing through this function.

    Returns: T statistic for the one sample test.
    '''

    cdef double t

    cdef double[:] xMatrix
    xMatrix = np.zeros((x.shape[0]), dtype=np.double)

    xMatrix = floatConverter1DArray(x)

    t = (np.mean(xMatrix) - popMean) / (np.sqrt(GetVariance(xMatrix))/np.sqrt(xMatrix.shape[0]))

    return t


def GetTwoSampleTTest(fusedDType[:] x1, fusedDType2[:] x2, xVariables=[], equalVariance=True):
    '''This function calculates the two sample test for determining if the means of two 
    populations are equal.

    Parameters:

    x1: First sample for two sample t test. Must convert objects to a list or numpy   
        array before passing through.

    x2: Second sample for two sample t test. Must convert objects to a list or numpy   
        array before passing through this function.

    Optional Parameters:

    xVariables: List of the independent variable names in the order they were passed into the model.

    equalVariance: Boolean indicator for assumption about equal variance between two samples.

    Returns: Dictionary with variables and the t statistic.
    '''

    cdef double[:] xMatrix1
    xMatrix1 = np.zeros((x1.shape[0]), dtype=np.double)

    cdef double[:] xMatrix2
    xMatrix2 = np.zeros((x2.shape[0]), dtype=np.double)

    xMatrix1 = floatConverter1DArray(x1)
    xMatrix2 = floatConverter1DArray(x2)

    cdef dict twoSampleDict
    cdef double pooledSE, t, posOne, posTwo
    posOne=1
    posTwo=2

    twoSampleDict = {}

    if len(xVariables) == 0:
        xVariables = ['x_1','x_2']
    else:
        xVariables = xVariables
    
    xKey = ' and '.join(xVariables)

    if equalVariance == True:
        pooledSE = np.sqrt((((xMatrix1.shape[0]-posOne)*GetVariance(xMatrix1))+((xMatrix2.shape[0]-posOne)*GetVariance(xMatrix2)))/(xMatrix1.shape[0]+xMatrix2.shape[0]-posTwo))
        t = (np.mean(xMatrix1) - np.mean(xMatrix2))/(pooledSE*np.sqrt(posOne/xMatrix1.shape[0]+posOne/xMatrix2.shape[0]))
    else:
        t = (np.mean(xMatrix1) - np.mean(xMatrix2))/np.sqrt((GetVariance(xMatrix1)/xMatrix1.shape[0]+(GetVariance(xMatrix2)/xMatrix2.shape[0])))

    twoSampleDict[str(xKey) + ' t statistic:'] =  t

    return twoSampleDict