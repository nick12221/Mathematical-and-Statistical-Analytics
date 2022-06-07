'''
The MathematicalAndStatisticalAnalytics package provides:
1. Functions for performing linear algebra relevant to linear regression.
2. Functions for linear regression modelling.
3. OLS Model Performance Metrics.
4. New OLS variation for variable selection called Auto Linear Fit.

NumPy is needed for this package.

What is included in the documentation:

Docstrings for all functions are included. The docstrings can be accessed
using the help command.
'''


import numpy as np


######-----------------------------------------------------------Classes------------------------------------------------------------------######


class colorScheme:
   purple = '\033[95m'
   blue = '\033[94m'
   green = '\033[92m'
   yellow = '\033[93m'
   red = '\033[91m'
   bold = '\033[1m'
   underline = '\033[4m'
   resetFont = '\033[0m'


######--------------------------------------------------- Basic Math Functions -----------------------------------------------------------######


def GetProduct(x):
    '''This function returns the product of a list, series or array of numbers.
    
    Parameters:
    
    x: A list, array, or pandas series of numbers to be passed into the function. Values must be integers or floats.
    
    Returns: The product of the numbers.
    '''
    
    start = [1]

    for i in range(0,len(x)):
        start.append(start[i]*x[i])

    return start[-1]


def GetSquareRoot(x):
    '''This function calculates the square root for a single number or list of numbers.
    
    Parameters:
    
    x: Can be a numeric (integer or float) value or list of values. Numbers must be greater than zero.
    
    Returns: The square root of a single number or square root of all numbers in a list.
    '''
    
    if isinstance(x, (np.ndarray,list)):
        return [i**.5 for i in x]
    else:
        return x**.5


def GetMean(x):
    '''This function calculates the mean for a single number or list of numbers.
    
    Parameters:
    
    x: Can be a numeric (integer or float) value or list of values. 
    
    Returns: The mean of a single number or mean of all numbers in a list.
    '''
    
    if isinstance(x, (np.ndarray,list)):
        return sum(x)/len(x)
    else:
        return x


def GetSquared(x):
    '''This function calculates the square for a single number or list of numbers.
    
    Parameters:
    
    x: Can be a numeric (integer or float) value or list of values.
    
    Returns: The square of a single number or square of all numbers in a list.
    '''
    
    if isinstance(x, (np.ndarray,list)):
        return [i**2 for i in x]
    else:
        return x**2


def GetReciprocal(x):
    '''This function calculates the reciprocal for a single number or list of numbers. The reciprocal is defined as 1/number.
    
    Parameters:
    
    x: Can be a numeric (integer or float) value or list of values.
    
    Returns: The reciprocal of a single number or reciprocal of all numbers in a list.
    '''

    if isinstance(x, (np.ndarray,list)):
        return [1/i for i in x]
    else:
        return 1/i


def GetNaturalLog(x):
    '''This function calculates the natural log for a single number or list of numbers.
    This leverages numpy built in natural log function.
    
    Parameters:
    
    x: Can be a numeric (integer or float) value or list of values. Numbers must be greater than zero.
    
    Returns: The natural log of a single number or natural log of all numbers in a list.
    '''

    if isinstance(x, (np.ndarray,list)):
        return [np.log(i) for i in x]
    else:
        return np.log(i)


#####-------------------------------------- Predefined Dictionaries for Auto Linear Fit Model ----------------------------------------------#####


LinearTransformationDict = {'square':GetSquared,
                            'squareRoot':GetSquareRoot,
                            'reciprocal':GetReciprocal,
                            'naturalLog':GetNaturalLog}


#####-----------------------------------Linear Algebra and Functions to build OLS Regression Models-----------------------------------------#####


def GetMatrix(rows, columns, normalMean=0, normalStd=1):
    '''This function creates a matrix of n X m specified dimensions with a random number generator that 
    follows a normal distribution.
    
    Parameters:
    
    rows: An integer value for the number of rows in a matrix. 
    
    columns: An integer value for the number of columns in a matrix.

    Optional Parameters:
    
    normalMean: Integer or float value for the mean of the randomly generated variable that follows a normal
                distribution. The default value is centered at a mean of 0.
    
    normalStd: Integer or float value for the standard deviation of the randomly generated variable that follows a 
               normal distribution. The default value is 1.
    
    Returns: Matrix of specified dimensions.
    '''
    
    assert isinstance(rows, int), "Number of rows must be an integer."
    assert isinstance(columns, int), "Number of columns must be an integer."  
    
    matrix = []
    
    for i in range(0,rows):
        arrayList = []
        for i in range(0, columns):
            x = np.random.normal(normalMean, normalStd, size=1)
            arrayList.append(x[0])
        
        matrix.append(arrayList)
    
    return matrix


def GetIdentityMatrix(dimensions):
    '''This function creates an identity matrix of n X n dimensions. An identity matrix has 1 across its diagonal
    and zeroes in all other locations.
    
    Parameters:
    
    dimensions: An integer value for the dimensions of the n X n square matrix. 
    
    Returns: An identity matrix of n X n dimensions.
    '''
    
    assert isinstance(dimensions, int), "Number of dimensions must be an integer."
    
    identityMatrix = [[0]*dimensions]*dimensions
    
    for i in range(dimensions):
        identityMatrix[i][i] = 1
                
    return identityMatrix


def GetTransposeMatrix(matrix1):
    '''This function transposes the rows into columns for a given matrix.
    
    Parameters: 
    
    matrix1: A n X m matrix or vector. 
    
    Returns: A transposed version of the original matrix or vector passed through the function.
    '''
    
    transposedMatrix = []
    
    for i in range(0,len(matrix1[0])):
        transposedMatrix.append([row[i] for row in matrix1])
        
    return transposedMatrix


def GetDotProduct(matrix1, matrix2):
    '''This function calculates the dot product between two matrices. 
    
    Parameters:
    
    matrix1: This is a matrix whose n X m dimensions are integer values. These can be any dimensions, as long as the
             number of columns in this matrix equals the number of rows in the second matrix.
    
    matrix2: This is a matrix whose m X p dimensions are integer values. These can be any dimensions, as long as the
             number of rows in this matrix equals the number of columns in the first matrix.
    
    Returns: A square matrix of m X m dimensions that is the result of the dot product calculated from the first and
             second matrix.
    '''
    
    assert len(matrix1[0]) == len(matrix2), "The number of columns in matrix 1 must match the rows in matrix 2."
    
    zipMatrix2 = zip(*matrix2)
    zipMatrix2 = list(zipMatrix2)
    multMatrix = [[sum(map(lambda x, y: x * y, rowA, colB)) for colB in zipMatrix2] for rowA in matrix1]
        
    return multMatrix


def GetMatrixDiagonal(matrix1):
    '''This function extracts the diagonal of a square matrix that can have any n X n dimensions.
    
    Parameters:
    
    matrix1: An n X n matrix that can be of any size. 
    
    Returns: A vector that is the diagonal of the matrix passed through the function.
    '''
    
    assert len(matrix1[0]) == len(matrix1), "Must pass a square matrix."
    
    diagonalMatrix = [matrix1[i][i] for i in range(len(matrix1))]
        
    return diagonalMatrix


def GetDeepCopy(lst):
    '''This function returns a deep copy of a list or list of lists.
    
    Parameters: 
    
    lst: List to make a deep copy of.
    
    Returns: Copy of a list or list of lists.
    '''
    
    if type(lst[0]) != list:
        return [x for x in lst]
    else:
        return [GetDeepCopy(lst[x]) for x in range(len(lst))]


def GetDeterminant(matrix1):
    '''This function calculates the determinant of a square matrix.
    
    Parameters:
    
    matrix1: A square matrix.
    
    Returns: The determinant of the matrix passed into the function.
    '''
    
    assert len(matrix1[0]) == len(matrix1), "Must pass a square matrix."
    
    matrix2 = GetDeepCopy(matrix1)
    
    if sum([sub[0] for sub in matrix2]) == 0:
        return 0
    elif len(matrix1) == 1:
        return matrix1[0][0]
    else:
        rowChange = 0
        if matrix2[0][0] == 0:
            FirstColumnList = [sub[0] for sub in matrix2]
            rowChange = next(x for x, val in enumerate(FirstColumnList) if val > 0)
            matrix2[0], matrix2[rowChange] = matrix2[rowChange], matrix2[0]
            rowChange = 1
        for i in range(len(matrix2)):
            for j in range(i+1, len(matrix2)):
                if matrix2[i][i] == 0:
                    return 0
                RowScaler = matrix2[j][i] / matrix2[i][i]
                for k in range(len(matrix2)): 
                    matrix2[j][k] = matrix2[j][k] - matrix2[i][k] * RowScaler
                    
    determinant = (-1)**rowChange*GetProduct(GetMatrixDiagonal(matrix2))
    
    return determinant


def GetMatrixMinor(matrix1,i,j):
    '''This function calculates the minor of a specified matrix.
    
    Parameters:
    
    matrix1: Matrix to pass through the function.
    
    i: index of columns of matrix to include for the minor.
    
    j: index of rows of matrix to include for the minor.
    
    Returns: The matrix minor for the input matrix.
    '''
    
    return [row[:j] + row[j+1:] for row in (matrix1[:i]+matrix1[i+1:])]


def GetMatrixInverse(matrix1):
    '''This function calculates the inverse of a matrix. 
    
    Parameters:
    
    matrix1: An n X n square matrix.
    
    Returns: The inverse matrix of the matrix passed through the function.
    '''
    
    assert GetDeterminant(matrix1)> 0 or GetDeterminant(matrix1)< 0, "Matrix is singular and cannot be inverted."
    
    identityMatrix = GetIdentityMatrix(len(matrix1))
    determinant = GetDeterminant(matrix1)
    matrix2 = GetDeepCopy(matrix1)
    
    if len(matrix2) == 1:
        return [[1/matrix2[0][0]]]    

    if len(matrix2) == 2:
        return [[matrix2[1][1]/determinant, -1*matrix2[0][1]/determinant],
                [-1*matrix2[1][0]/determinant, matrix2[0][0]/determinant]]
    
    cofactors = []
    
    for r in range(len(matrix2)):
        cofactorRow = []
        for c in range(len(matrix2)):
            minor = GetMatrixMinor(matrix2,r,c)
            cofactorRow.append(((-1)**(r+c)) * GetDeterminant(minor))
            
        cofactors.append(cofactorRow)
        
    cofactors = GetTransposeMatrix(cofactors)
    
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
            
    return cofactors


def GetCoefs(y, x):
    '''This function calculates the beta coefficients for the OLS Model. 
    
    Parameters:
    
    y: Response variable to pass through the model. Can pass a numpy array or list object. Dataframes 
       must be converted to an array or list before being passed through the function.

    x: X variables to pass through the model. Can pass a numpy array or list object. Dataframes 
       must be converted to an array or list before being passed through the function.

    Returns: List of the OLS model coefficients.
    '''
    
    assert len(x) == len(y), "X and Y data must be the same length."
    
    xTransposed = GetTransposeMatrix(x)
    coefficients = GetDotProduct(GetDotProduct(GetMatrixInverse((GetDotProduct(xTransposed, x))), xTransposed), y)

    return coefficients


def GetPredictions(coefs, x):
    '''This function calculates the predictions for a given dataset. 
    
    Parameters:
    
    coefs: Coefficients from the fitted model. Can pass a numpy array or list object. Dataframes 
       must be converted to an array or list before being passed through the function.

    x: X variables to pass through the model. Can pass a numpy array or list object. Dataframes 
       must be converted to an array or list before being passed through the function.
    
    Returns: List of predictions.
    '''
    
    preds = GetDotProduct(GetTransposeMatrix(coefs), GetTransposeMatrix(x))
    
    if type(preds[0]) == list:
        predictions = preds[0]
    else:
        predictions = preds
    
    return predictions


def GetCoefStdErrors(y, x, coefs):
    '''This function calculates the standard errors for the beta coefficients. 
    
    Parameters:
    
    y: Response variable to pass through the model. Can pass a numpy array or list object. Dataframes 
       must be converted to an array or list before being passed through the function.

    x: X variables to pass through the model. Can pass a numpy array or list object. Dataframes 
       must be converted to an array or list before being passed through the function.

    coefs: Coefficients from the model you want to use for prediction. Can pass a numpy array or list object. 
           Dataframes must be converted to an array or list before being passed through the function.
    
    Returns: List of standard errors of the coefficients.
    '''
    
    assert len(x) == len(y), "X and Y data must be the same length."
    
    invertedMatrix = GetMatrixInverse(GetDotProduct(GetTransposeMatrix(x), x))
    diagonalMatrix = GetMatrixDiagonal(invertedMatrix)
    mse = GetMSE(y, GetPredictions(coefs, x))
    stdErrorMatrix = list((mse*diagonalMatrix)**.5)
    
    return stdErrorMatrix


def GetVif(x):
    '''This function calculates the vif scores for all x variables. Vif is a measure for multicollinearity.
    This function is dependent on the GetTransposeMatrix, GetPredictions and GetR2 functions.
    
    Parameters:
    
    x: A matrix of independent variables that are used in a specified OLS model. Each variable will be used as the 
       response variable with the others used as predictors to calculate the vif scores.
    
    Returns: A list of vif scores for each independent variable.
    '''
    
    vifScores = []
    
    for i in range(len(x[0])):
        y3 = [[sub[i] for sub in x]]
        xNew = [[sub[j] for j in range(len(x[0])) if j !=i] for sub in x]
        y4 = GetTransposeMatrix(y3)
        R2 = GetR2(y3[0], GetPredictions(GetCoefs(y4, xNew), xNew))
        vif = 1/(1-R2)
        vifScores.append(vif)
    
    return vifScores


def GetOLSResults(y, x, xVariables=[]):
    '''This function calculates the OLS equation for the given y and x variables. A summary of the
    model is printed and a dictionary is created to store all OLS results. 
    
    Parameters:
    
    y: Response variable to pass through the model. Can pass a numpy array or list object. Dataframes 
       must be converted to an array or list before being passed through the function.

    x: X variables to pass through the model. Can pass a numpy array or list object. Dataframes 
       must be converted to an array or list before being passed through the function.

    Optional Parameters:
    
    xVariables: List of the x variable names in the order they were passed into the model.
    
    Returns: A dictionary with coefficients, standard errors, t value, and variable names.
    '''
    
    modelCoefs = GetCoefs(y, x)
    R2Value = GetR2(y, GetPredictions(modelCoefs,x))
    stdErrorCoefs = GetCoefStdErrors(y, x, modelCoefs)
    modelCoefs = GetTransposeMatrix(modelCoefs)
    coefficients = modelCoefs[0]
    
    tValues = [coefficients[i] / stdErrorCoefs[i] for i in range(len(modelCoefs[0]))]
    
    if len(xVariables) > 0:
        xVariables = xVariables
    else:
        xVariables = ['x_' + str(i) for i in range(len(modelCoefs[0]))]
    
    results = {'Variables': xVariables,
               'Model Coefficients': coefficients,
               'Coef Std. Errors': stdErrorCoefs,
               'T Stat':tValues
               }

    print(colorScheme.purple+colorScheme.bold+colorScheme.underline+'Model Summary Statistics'+colorScheme.resetFont)
    print(f'Number of Observations: {len(x)}')
    print(f'Number of DoF: {len(x)-len(x[0])}')
    print(f'R2: {round(R2Value,5)}')
    print(f'Adjusted R2: {round(GetAdjustedR2(R2Value, len(x), len(x[0])),5)}')

    return results


def GetAltOLS(y, x, xVariables=[], transformationDict=LinearTransformationDict, R2PerformanceThreshold = .1):
    '''This function is used for variable selection. Instead of assuming a linear relationship
    between the independent and response variables, this function tests if non linear relationships
    fit the data better. An auto fit is done for each variable and the transformation with the highest 
    R2 score is selected. Categoricals must be dummied before passing data through the model.

    Parameters: 

    y: Response variable to pass through the model. Can pass a numpy array or list object. Dataframes 
       must be converted to an array or list before being passed through the function.

    x: X variables to pass through the model. Can pass a numpy array or list object. Dataframes 
       must be converted to an array or list before being passed through the function.

    Optional Parameters:
    
    xVariables: List of the x variable names in the order they were passed into the model.

    transformationDict: A dictionary containing the name of the transformation and the function
                        containing the type of linear transformation to perform.

    R2PerformanceThreshold: Minimum increase in R2 a linear transformation must achieve to
                            be chosen over the default selection of a linear relationship.

    Returns: A dictionary with the results of the AltOLS model.
    '''

    modelCoefs = GetCoefs(y, x)
    R2Benchmark = GetR2(y, GetPredictions(modelCoefs, x))
    xFinal = GetTransposeMatrix(GetDeepCopy(x))

    if len(xVariables) == 0:
        xVariables = ['x_'+str(i) for i in range(len(x[0]))]
    else:
        xVariables = xVariables

    for i in range(len(x[0])):
        xTranposedMatrix = GetTransposeMatrix(x)
        uniqueValuesX = set(xTranposedMatrix[i])
        
        if len(uniqueValuesX) <= 2:
            continue

        R2ScoresDict = {
                        }
        
        for key, value in transformationDict.items():
            xCopy = GetDeepCopy(xTranposedMatrix)

            if key == 'square':
                xCopy.append(value(xCopy[i]))
            elif (key in ['reciprocal','naturalLog']) & (any([v == 0 for v in xCopy[i]]) == True):
                continue
            elif (key == 'squareRoot') & (any([v < 0 for v in xCopy[i]]) == True):
                continue
            else:
                xCopy.append(value(xCopy[i]))
                xCopy.remove(xCopy[i])
            
            R2ScoresDict[key] = GetR2(y, 
                                GetPredictions(GetCoefs(y, GetTransposeMatrix(xCopy)), GetTransposeMatrix(xCopy)))

        maxValue = max(R2ScoresDict.values())
        maxKey = max(R2ScoresDict, key=R2ScoresDict.get)

        if maxValue - R2Benchmark > R2PerformanceThreshold:
            if maxKey == 'square':
                xFinal.append(LinearTransformationDict[maxKey](xTranposedMatrix[i]))
                xVariables.append("".join([str(xVariables[i]),'_',maxKey]))
            else:
                xFinal.append(value(xTranposedMatrix[i]))
                xFinal.remove(xTranposedMatrix[i])
                xVariables.append("".join([str(xVariables[i]),'_',maxKey]))
                xVariables.remove(xVariables[i])
        else:
            continue

    return GetOLSResults(y, GetTransposeMatrix(xFinal), xVariables)
            

#####--------------------------------------- Linear Regression Metrics and Statistical TestingFunctions ------------------------------------------#####


def GetVariance(numbers):
    '''This function calculates the variance of a list or array of numbers.
    
    Parameters:
    
    numbers: The numbers you want to calculate the variance for. Can pass a list, array, or pandas series.
    
    Returns: Variance of the numbers that were passed through the function.
    '''
    
    mean = (sum(numbers)/len(numbers))
    total = sum([(numbers[i] - mean)**2 for i in range(len(numbers))])
    variance = total/(len(numbers)-1)
    
    return variance


def GetR2(yTrue, yPred):
    '''This function calculates the R2 for a given Ordinary Least Squares model. R2 is the amount of variance
    explained by the linear regression model. It can be negative and cannot take a value higher than 1. 
    
    Parameters:
    
    yTrue: Actual y values that were used to train the OLS model. yTrue can be passed as a list,
           array, or a pandas series. Must be integer or float values.
    
    yPred: Predicted y values that were calculated using the trained OLS model. yPred can be
           passed as a list, array, or a pandas series. Must be integer or float values.
    
    Returns: R2 value of a given model.
    '''
    
    assert len(yTrue) == len(yPred), "Actual and Predicted Y arrays must be the same length."
    
    squaredDifferences = [(i-j)**2 for i, j in zip(yTrue, yPred)]
    yMean = sum(yTrue)/len(yTrue)
    totalVariability = [(i - yMean)**2 for i in yTrue]

    if sum(totalVariability) == 0:
        return np.nan
    else:
        r2 = 1 - sum(squaredDifferences)/sum(totalVariability)
    
    if isinstance(r2, (np.ndarray,list)):
        return r2[0]
    else:
        return r2


def GetMSE(yTrue, yPred):
    '''This function calculates the Mean Squared Error for a given Ordinary Least Squares model. 
    
    Parameters:
    
    yTrue: Actual y values that were used to train the OLS model. yTrue can be passed as a list,
           array, or a pandas series. Must be integer or float values.
    
    yPred: Predicted y values that were calculated using the trained OLS model. yPred can be
           passed as a list, array, or a pandas series. Must be integer or float values.
    
    Returns: The MSE of a given model.
    '''
    
    assert len(yTrue) == len(yPred), "Actual and Predicted Y arrays must be the same length."
    
    squaredError = [(i-j)**2 for i, j in zip(yTrue, yPred)]
    MSE = sum(squaredError)/len(yTrue)

    return MSE


def GetAdjustedR2(R2, n, p):
    '''This function calculates the Adjusted R2 of a regression output. The p parameter 
    for number of predictors should include the intercept in the count.
    
    Parameters:
    
    R2: The R2 score from the model.
    
    n: Sample size in the corresponding OLS Model. Must be an integer.
    
    p: Number of predictors used in the OLS model including the intercept. Must be an integer.
    
    Returns: The Adjusted R2 score.
    '''
    
    adjr2 = 1 - ((1 - R2)*(n-1))/(n-p)
    
    return adjr2


def GetPearsonCorr(x, y):
    '''This function calculates the pearson correlation coefficient between two variables.
    
    Parameters:
    
    x: First variable to pass to the function. Can be a list, array, or pandas series.
    
    y: Second variable to pass to the function. Can be a list, array, or pandas series.
    
    Returns: One number indicating the correlation coefficient between two variables.
    '''
    
    assert len(x) == len(y), "X and Y vectors must be the same length."
    
    xMean = sum(x)/len(x)
    yMean = sum(y)/len(y)
    numerator = [(i-xMean)*(j-yMean) for i, j in zip(x, y)]
    xSpread = [(i-xMean)**2 for i in x]
    ySpread = [(i-yMean)**2 for i in y]
    r = sum(numerator)/(sum(xSpread)*sum(ySpread))**.5
    
    return r


def GetTwoSampleTTest(x1, x2, xVariables=[], equalVariance=True):
    '''This function calculates the two sample test for determining if the means of two 
    populations are equal.

    Parameters:

    x1: First population for two sample t test. Please convert objects to a list or numpy   
        array for this function.

    x2: Second population for two sample t test. Please convert objects to a list or numpy   
        array for this function.

    Optional Parameters:

    xVariables: List of the x variable names in the order they were passed into the model.

    equalVariance: Boolean indicator for assumption about equal variance between two populations.

    Returns: Dictionary with variables and the t stat.
    '''

    twoSampleDict = {}
    
    if len(xVariables) == 0:
        xVariables = ['x_1','x_2']
    else:
        xVariables = xVariables
        
    xKey = ' and '.join(xVariables)
    
    if equalVariance == True:
        pooledSE = GetSquareRoot((((len(x1)-1)*GetVariance(x1))+((len(x2)-1)*GetVariance(x2)))/(len(x1)+len(x2)-2))
        t = (GetMean(x1) - GetMean(x2))/(pooledSE*GetSquareRoot(1/len(x1)+1/len(x2)))
    else:
        t = (GetMean(x1) - GetMean(x2))/ GetSquareRoot((GetVariance(x1)/len(x1)) + (GetVariance(x2)/len(x2)))
    
    twoSampleDict[xKey] = 't stat: ' + str(t)

    return twoSampleDict