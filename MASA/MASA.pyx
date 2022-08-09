'''
The MathematicalAndStatisticalAnalytics package provides:
1. Functions for performing linear algebra relevant to OLS regression.
2. Class for OLS regression modelling and new variation called Auto Linear Fit.
3. Functions for OLS model performance metrics.
4. Functions for statistics and statistical testing.


NumPy is needed for this package. All functions take numpy ndarray type objects of either 1 or 2 dimensions.
Please refer to each function's documentation to check array dimension specifications. This package is 
written in Cython, primarily relying on variable declaration and memory views for efficiency gains. Set up  
file is included and needed to run this package.

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


###-------------------------------------OLS Lin Alg Functions-------------------------------------###
                #####################################################################
                #####################################################################
                #####################################################################
                #####################################################################
                ##########  All Linear Algebra functions accept 2d numpy  ###########
                ##########  arrays of either float or integer values.     ###########
                #####################################################################
                #####################################################################
                #####################################################################
                #####################################################################

def GetMatrix(long long int rows, long long int columns):
    '''This function creates a numpy ndarray of n X m specified dimensions. The numpy 
    random number generator is used for generating datapoints.

    Parameters:

    rows: Number of rows in the numpy array. 

    columns: Number of columns in the numpy array.

    Returns: A 2d numpy ndarray of specified dimensions with float datatype values.
    '''

    result = np.zeros((rows, columns))

    cdef Py_ssize_t i, j

    for i in range(rows):
        for j in range(columns):
            x = np.random.rand()  
            result[i, j] = x

    return result


def floatConverter(fusedDType[:,:] intMatrix):
    '''This function converts the dataype of the input matrix to a float.
    A cython defined fused datatype accepts either float or int datatypes,
    and then are all converted to floats.

    Parameters:

    intMatrix: A matrix that has a c defined fused datatype that accepts
               either ints or floats. A 2d numpy array must be passed
               through this function.

    Returns: A numpy 2d array that has values of float datatype.
    '''

    cdef double[:,:] floatMatrix = np.array(intMatrix, dtype=np.double)

    return floatMatrix


def floatConverter1DArray(fusedDType[:] intMatrix):
    '''This function converts the dataype of the input matrix to a float.
    A cython defined fused datatype accepts either float or int datatypes,
    and then are all converted to floats.

    Parameters:

    intMatrix: A matrix that has a c defined fused datatype that accepts
               either ints or floats. A 1d numpy array must be passed
               through this function.

    Returns: A numpy 1d array that has values of float datatype.
    '''

    cdef double[:] floatMatrix = np.array(intMatrix, dtype=np.double)

    return floatMatrix


def GetIdentityMatrix(long long int dimensions):
    '''This function creates an identity matrix of n X n dimensions. An identity matrix is
    a square matrix that has 1 across its diagonal and zeroes in all other locations. 

    Parameters:

    dimensions: The dimensions of the n X n square matrix. 

    Returns: A numpy 2d array identity matrix of n X n dimensions.
    '''
    
    identityMatrix = np.zeros((dimensions, dimensions), dtype=np.double)

    cdef Py_ssize_t i

    for i in range(dimensions):
        identityMatrix[i, i] = 1
            
    return identityMatrix


def GetTransposeMatrix(fusedDType[:, :] matrix1):
    '''This function transposes the rows into columns for a given matrix. The data in the 
    matrix can be of int or float datatype.

    Parameters: 

    matrix1: A 2d numpy array that can have values of float or int datatypes. 

    Returns: A transposed version of the original matrix with values of float datatype.
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
    '''This function calculates the dot product of two matrices. The dot product is the 
    product of the magnitude of the two vectors and their cosine angle. Two cython defined
    fused datatypes are used here for the two matrices. Each can take int or float values.
    
    Parameters:
    
    matrix1: A 2d numpy array that can have values of float or int datatypes. 
              The number of columns in matrix1 must equal the number of rows in matrix2.
    
    matrix2: A 2d numpy array that can have values of float or int datatypes.  
             The number of rows in matrix2 must equal the number of columns in matrix1.
    
    Returns: A square 2d numpy array with values of float datatype.
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
    '''This function extracts the diagonal of a square matrix.

    Parameters:

    matrix1: A 2d numpy array that can have values of float or int datatypes. 

    Returns: A 2d numpy array with values of float datatype containing the 
             values along the diagonal of the matrix passed into the function.
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

    matrix1: A square 2d numpy array that can have values of float or int
             datatypes.

    Returns: The determinant as a float datatype for the matrix passed to the function.
    '''
    
    matrixLength = matrix1.shape[0]

    cdef fusedDType[:, :] matrix
    matrix = np.zeros((matrixLength, matrixLength), dtype=np.double)

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

    matrix1: A square 2d numpy array that can have values of float or int
             datatypes.

    i: index of row of matrix to exclude for the minor.

    j: index of column of matrix to exclude for the minor.

    Returns: A 2d numpy array that is the minor of the input matrix.
    '''

    matrixMinor = np.delete(matrix1, i, 0)
    matrixMinor = np.delete(matrixMinor, j, 1)

    return matrixMinor


def GetMatrixInverse(fusedDType[:, :] matrix1):
    '''This function calculates the inverse of a matrix. 

    Parameters:

    matrix1: A square 2d numpy array that can have values of float or int
             datatypes.

    Returns: A 2d numpy array that is the inverse of the matrix passed through the function
             with values of float dataype.
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

    cofactorMatrix = cofactorMatrix/determinant
        
    return cofactorMatrix


###----------------------------------------OLS Model Class----------------------------------------###


class OLSSuite:
                
                #####################################################################
                #####################################################################
                #####################################################################
                #####################################################################
                ##########  All OLSSuite methods accept 2d numpy arrays   ###########
                ##########  of either float or integer values.            ###########
                #####################################################################
                #####################################################################
                #####################################################################
                #####################################################################

    '''This class is designed for building OLS model objects. OLSSuite is built on top of numpy arrays,
    leveraging the ndarray object for matrix operations. All objects passed through this class' methods 
    must be 2d numpy arrays. Functions are all written in Cython for more efficient matrix operations. 

    Methods:

    GetOLSResults: This method fits the model object with the OLS results of the specified independent 
                   and dependent variables. All ten attributes are populated, with the primary results  
                   being stored in the OLSResults and Metrics dictionary. Two dimensional arrays must 
                   be passed for the x and y variables.

    GetAltOLS: This method is a variation of the standard OLS methodology. Instead of assuming all 
               independent variables have a linear relationship with the dependent variable, this 
               algorithm tests non linear relationships. If there is an improvement in R2 greater  
               than the benchmark rate, that transformation will automatically be applied and OLS model 
               results from the auto fitted model will be returned.

    GetPredictions: Method for making predictions based on the fitted model object. Model object must 
                    first be fit using GetOLSResults or GetAltOLS methods before making predictions.

    Attributes: 

    OLSResult: A dictionary of the regression results.

    Coefficients: Beta coefficients for the fitted model.

    stdErrorCoefs: The standard errors of the model coefficients.

    R2Score: R2 score of the model.
    
    AdjR2Score: Adjusted R2 score of the model.

    dofResiduals: The degrees of freedom of the residuals of the model.

    nObservations: The number of observations in the OLS model.

    meanAbsoluteError: The mean absolute error of the OLS model.
    '''

    def __init__(self):
        '''This function initializes ten attributes for the OLSSuite class. Description of attributes
        are provided in the class description.
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
        '''This method calculates the beta coefficients and their standard errors for the OLS Model. 
    
        Parameters:
    
        y: A 2d numpy array of the response variable for the model. Please convert list of lists or 
           Dataframes to numpy arrays before passing objects through the model.

        x: A 2d numpy array of the independent variables for the model. Please convert list of lists
           or Dataframes to numpy arrays before passing objects through the model.

        Returns: The coefficients and standard errors attributes that are a 2d numpy array from the 
        fitted model object.
        '''

        cdef double errorModelCoefs
        cdef Py_ssize_t i, j
        j = 0
        
        cdef double[:, :] xMatrix
        xMatrix = np.zeros((x.shape[0], x.shape[1]), dtype=np.double)

        cdef double[:, :] yMatrix
        yMatrix = np.zeros((y.shape[0], y.shape[1]), dtype=np.double)

        squaredError = np.zeros((y.shape[0], y.shape[1]), dtype=np.double) 
        cdef double[:, :] resultView = squaredError

        yMatrix = floatConverter(y)
        xMatrix = floatConverter(x)

        xTransposed = GetTransposeMatrix(xMatrix)
        
        invertedMatrix = GetMatrixInverse((GetDotProduct(xTransposed, xMatrix)))
        self.Coefficients = GetDotProduct(GetDotProduct(invertedMatrix, xTransposed), yMatrix)

        diagonalMatrix = GetMatrixDiagonal(invertedMatrix)
        predMatrix = self.GetPredictions(xMatrix)

        for i in range(x.shape[0]):
            resultView[i, j] = (yMatrix[i, j] - predMatrix[i, j])**2

        errorModelCoefs = np.sum(squaredError)/(xMatrix.shape[0] - xMatrix.shape[1])

        self.stdErrorCoefs = (errorModelCoefs*diagonalMatrix)**.5


    def GetPredictions(self, fusedDType[:, :] x):
        '''This method calculates the predictions for a given dataset. The model must be fit using the
        GetOLS method first.
    
        Parameters:

        x: A 2d numpy array of the independent variables for the model. Please convert list of lists or 
           Dataframes to numpy arrays before passing objects through the model.
    
        Returns: A 2d numPy array of the predictions from the OLS model object.
        '''

        cdef double[:, :] xMatrix
        xMatrix = np.zeros((x.shape[0], x.shape[1]), dtype=np.double)

        xMatrix = floatConverter(x)
    
        preds = GetDotProduct(GetTransposeMatrix(self.Coefficients), GetTransposeMatrix(xMatrix))
        predictions = np.reshape(preds, (x.shape[0], 1))
    
        return predictions


    def GetOLSResults(self, fusedDType[:, :] y, fusedDType2[:, :] x, xVariables=[]):
        '''This function calculates the OLS equation for the given dependent and independent variables.
        Categorical variables must be dummied before passing data through the model. Must add constant 
        before running model.
        
        Parameters:
    
        y: A 2d numpy array of the response variable for the model. Please convert list of lists or 
           Dataframes to numpy arrays before passing objects through the model.

        x: A 2d numpy array of the independent variables for the model. Please convert list of lists or 
           Dataframes to numpy arrays before passing objects through the model.

        Optional Parameters:
    
        xVariables: List of the independent variable names in the order they were passed into the model.
                    If no value is passed, arbitrary names of x0 through number of variables are used.
    
        Returns: All attributes of the OLSSuite class. The primary attributes of this function are the OLSResults,
                 which is a dictionary with the OLS model results, and the Metrics dictionary which contains
                 model evaluation metrics.
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
        predictions = self.GetPredictions(xMatrix)
        self.R2Score = GetR2(yMatrix, predictions)
        self.AdjR2Score = GetAdjustedR2(self.R2Score, x.shape[0], x.shape[1])
        self.dofResiduals = x.shape[0] - x.shape[1]
        self.MeanAbsError = GetMeanAbsError(yMatrix, predictions)
        self.mse = GetMSE(yMatrix, predictions)
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
                   'Coefficients': coefficients,
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


    def GetAltOLS(self, fusedDType[:, :] y, fusedDType2[:, :] x, xVariables=[], double R2PerformanceThreshold = .1):
        '''This function is used for determining best linear fit for each independent variable. Instead of
        assuming a linear relationship between the independent and response variables, this function tests 
        if non linear relationships fit the data better. An auto fit is done for each variable and the 
        transformation with the highest R2 score is selected. Categorical variables must be dummied before  
        passing data through the model. Must add constant before running model.

        Parameters: 

        y: A 2d numpy array of the response variable for the model. Please convert list of lists or 
           Dataframes to numpy arrays before passing objects through the model.

        x: A 2d numpy array of the independent variables for the model. Please convert list of lists
           or Dataframes to numpy arrays before passing objects through the model.

        Optional Parameters:
    
        xVariables: List of the independent variable names in the order they were passed into the model.
                    If no value is passed, arbitrary names of x0 through number of variables are used.

        R2PerformanceThreshold: A float value that is the minimum increase in R2 a linear transformation must 
                                achieve to be chosen over the default assumption of a linear relationship.

        Returns: All attributes of the OLSSuite class, fitted using the auto linear fit methodology. The 
                 primary attributes of this function are the OLSResults, which is a dictionary with the 
                 OLS model results, and the Metrics dictionary which contains model evaluation metrics.
        '''

        cdef Py_ssize_t i
        cdef dict R2ScoresDict, LinearTransformationDict
        cdef double maxValue, R2Benchmark, floatZero
        cdef str maxKey

        floatZero = 0

        LinearTransformationDict = {'square': np.square,
                            'squareRoot': np.sqrt,
                            'reciprocal': np.reciprocal,
                            'naturalLog': np.log}

        cdef double[:, :] xMatrix
        xMatrix = np.zeros((x.shape[0], x.shape[1]), dtype=np.double)

        cdef double[:, :] yMatrix
        yMatrix = np.zeros((y.shape[0], y.shape[1]), dtype=np.double)

        altXMatrix = np.zeros((x.shape[0], x.shape[1]), dtype=np.double)

        yMatrix = floatConverter(y)
        xMatrix = floatConverter(x)

        self.GetOLSResults(yMatrix, xMatrix)
        R2Benchmark = self.R2Score

        if len(xVariables) == 0:
            xVariables = ['x_'+str(i) for i in range(xMatrix.shape[0])]

        xFinal = xMatrix.copy()

        for i in range(xMatrix.shape[1]):
            uniqueValsList = np.unique(xMatrix[:, i])

            if uniqueValsList.shape[0] <= 2:
                continue

            R2ScoresDict = {
                            }
        
            for key, value in LinearTransformationDict.items():
                altXMatrix = xMatrix.copy()

                if key == 'square':
                    altXMatrix = np.append(altXMatrix, value(np.reshape(altXMatrix[:, i], (-1, 1))), 1)
                elif (key in ['reciprocal','naturalLog']) & (any([v == 0 for v in altXMatrix[:, i]]) == True):
                    continue
                elif (key == 'squareRoot') & (any([v < 0 for v in altXMatrix[:, i]]) == True):
                    continue
                else:
                    altXMatrix = np.append(altXMatrix, value(np.reshape(altXMatrix[:, i], (-1, 1))), 1)
                    altXMatrix = np.delete(altXMatrix, i, 1)

                self.GetOLSResults(yMatrix, altXMatrix)
            
                R2ScoresDict[key] = self.R2Score

            maxValue = max(R2ScoresDict.values())
            maxKey = max(R2ScoresDict, key=R2ScoresDict.get)

            
            if maxValue - R2Benchmark > R2PerformanceThreshold:
                if maxKey == 'square':
                    xFinal = np.append(xFinal, LinearTransformationDict[maxKey](np.reshape(xMatrix[:, i], (-1, 1))), 1)
                    xVariables.append("".join([str(xVariables[i]),'_',maxKey]))
                else:
                    xFinal = np.append(xFinal, value(np.reshape(xMatrix[:, i], (-1, 1))), 1)
                    xFinal = np.delete(xFinal, i, 1)
                    xVariables.append("".join([str(xVariables[i]),'_',maxKey]))
                    xVariables.remove(xVariables[i])
            else:
                continue

        self.GetOLSResults(yMatrix, xFinal, xVariables)


###----------------------------------Regression Metrics Functions---------------------------------###


                #####################################################################
                #####################################################################
                #####################################################################
                #####################################################################
                ##########  Regression Metrics functions accept 2d numpy  ###########
                ##########  arrays of either float or integer values.     ###########
                #####################################################################
                #####################################################################
                #####################################################################
                #####################################################################


def GetMSE(fusedDType[:, :] yTrue, fusedDType2[:, :] yPred):
    '''This function calculates the Mean Squared Error for a given Ordinary Least Squares model. 

    Parameters:

    yTrue: A 2d numpy array containing the actual y values that were used to train the OLS model. 

    yPred: A 2d numpy array containing the predicted y values from the trained OLS model.

    Returns: The MSE of float dataype of a given model.
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
    explained by the linear regression model. 

    Parameters:

    yTrue: A 2d numpy array containing the actual y values that were used to train the OLS model. 

    yPred: A 2d numpy array containing the predicted y values from the trained OLS model.

    Returns: R2 value of float dataype of a given model.
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

    r2: The R2 score from the model. Must be a float.

    n: Sample size in the corresponding OLS Model. Must be a float.

    p: Number of predictors used in the OLS model including the intercept. Must be a float.

    Returns: The Adjusted R2 score of float datatype of a given model.
    '''

    cdef double posOne
    posOne = 1

    adjR2 = posOne - ((posOne - r2)*(n-posOne))/(n-p)

    return adjR2


def GetMeanAbsError(fusedDType[:, :] yTrue, fusedDType2[:, :] yPred):
    '''This function calculates the Mean Absolute Error for a given Ordinary Least Squares model. 

    Parameters:

    yTrue: A 2d numpy array containing the actual y values that were used to train the OLS model. 

    yPred: A 2d numpy array containing the predicted y values from the trained OLS model.

    Returns: The MAE of float datatype of a given model.
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
    '''This function calculates the vif scores, a measure of multicollinearity for all independent variables.

    Parameters:

    x: A 2d numpy array independent variables that are used in a specified OLS model. Each variable will
       be used as the response variable with the others used as predictors to calculate the vif scores.

    Returns: A 1d numpy aray of vif scores for each independent variable. Constant does not get a vif score.
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


                #####################################################################
                #####################################################################
                #####################################################################
                #####################################################################
                ##########  Statistical Testing functions accept 1d numpy  ##########
                ##########  arrays of either float or integer values.      ##########
                #####################################################################
                #####################################################################
                #####################################################################
                #####################################################################


def GetVariance(fusedDType[:] numbers):
    '''This function calculates the variance for an array of numbers.

    Parameters:

    numbers: A 1d array of the data to calculate the variance for. Values can be of float or 
             int datatypes.

    Returns: Variance of float dataype of the data passed through the function.
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

    x: A 1d numpy array of the data of float or int datatype.

    y: A 1d numpy array of the data of float or int datatype.

    Returns: Pearson correlation coefficient of float datatype between two variables.
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
    statistically different than a given population mean. This function only calculates a
    two sided one sample t test.

    Parameters:

    x: A 1d numpy array of data to test if mean is different than hypothesized value. Values
       can be of float or int datatypes.

    popMean: Value of float datatype for the assumed population mean.

    Returns: T statistic for the two sided one sample test.
    '''

    cdef double t

    cdef double[:] xMatrix
    xMatrix = np.zeros((x.shape[0]), dtype=np.double)

    xMatrix = floatConverter1DArray(x)

    t = (np.mean(xMatrix) - popMean) / (np.sqrt(GetVariance(xMatrix))/np.sqrt(xMatrix.shape[0]))

    return t


def GetTwoSampleTTest(fusedDType[:] x1, fusedDType2[:] x2, xVariables=[], equalVariance=True):
    '''This function calculates the two sample test for determining if the means of two 
    populations are equal. This function only calculates a two sided two sample t test.

    Parameters:

    x1: A 1d numpy array of the data for the first variable of the two sample t test. Values can
        be of float or int datatypes.

    x2: A 1d numpy array of the data for the second variable of the two sample t test. Values can
        be of float or int datatypes.

    Optional Parameters:

    xVariables: List of the independent variable names in the order they were passed into the model.
                If no value is passed, arbitrary names of x1 and x2 are used.

    equalVariance: Boolean indicator for assumption about equal variance between two samples.

    Returns: Dictionary with variables and the t statistic of the two sided two sample t test.
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