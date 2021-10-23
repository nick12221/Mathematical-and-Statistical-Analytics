# # Math and Application of Multivariate Linear Regression Modelling

# In the first part of this notebook I go through the exercise of creating functions to create matrices, 
# transpose matrices, create an identity matrix, calculate the dot product, get the diagonal matrix and then run through the formula used to calculate coefficient regressions in multivariate statistical analysis. In the second part I go through an EDA process with a data file, and in
# the third part use the formulas created from the linear algebra section to compute a regression model from scratch
# and evaluate its performance.

# ### Linear Algebra

# ###### Create Functions
import numpy as np
from numpy import random

'''Function to create a matrix (n x k) where the probability and distribution of a variable is conditional on another
variable that follows a binomial distribution. This simulation of variables is used to populate an n x k matrix.'''
def create_matrix(rows, columns, normal_mean, normal_std):
    array_of_array_list = []
    for i in range(0,rows):
        array_list = []
        for i in range(0, columns):
            random_variable = random.binomial(1, .2)
            if random_variable == 1:
                x = random.normal(normal_mean, normal_std, size=1)
                array_list.append(x[0])
            else:
                x = random.normal(normal_mean/2, normal_std/2, 1)
                array_list.append(x[0])
        
        array_of_array_list.append(array_list)
    
    matrix = np.array(array_of_array_list)
    return matrix

#Create n x n identity matrix
def create_identity_matrix(rows, columns):
    array_of_array_list = []
    if rows == columns:
        array_of_list = [1] + (columns-1)*[0]
        array_of_array_list.append(array_of_list)
        
        for i in range(0, rows-1):
            array = [array_of_array_list[i][-1]] + array_of_array_list[i][:-1]
            array_of_array_list.append(array)
    else:
        raise Exception("Matrix Must be a of equal rows and columns")
        
    return array_of_array_list

#Create a function to transpose any n x k matrix
def transpose_matrix(matrix):
    
    list_of_transposed_list = []
    for i in range(0,len(matrix[0])):
        transposed_list = []
        for j in range(0,len(matrix)):
            transposed_number = matrix[j][i]
            transposed_list.append(transposed_number)
        
        list_of_transposed_list.append(transposed_list)
    
    transposed_matrix = np.array(list_of_transposed_list)
    
    return transposed_matrix

#Create a function to calculate the dot product between two matrices
def dot_product(matrix1, matrix2):
    multmatrix = []
    
    for i in range(0,len(matrix1)):
        row_list = []
        for j in range(0, len(matrix2[0])):            
            number = 0
            
            for h in range(0, len(matrix2)):
                number += matrix1[i][h] * matrix2[h][j]
            
            row_list.append(number)
            
        multmatrix.append(row_list)
    
    multmatrix = np.array(multmatrix)
        
    return multmatrix

def get_matrix_diagonal(matrix):
    diagonal_matrix = []
    column_index = 0
    
    for i in range(len(matrix)):
        diagonal_number = matrix[i][column_index]
        diagonal_matrix.append(diagonal_number)
        column_index += 1
    
    diagonal_matrix = np.array(diagonal_matrix)
    return diagonal_matrix

#Use the functions from above as well as numpy's inverse function to calculate regression coefficients
def regression_formula(x_matrix, y_matrix):  
    transposed_xmatrix = transpose_matrix(x_matrix)
    identity_matrix = create_identity_matrix(len(x_matrix), len(transposed_xmatrix[0]))
    
    coefficient_matrix = dot_product(dot_product(np.linalg.inv((dot_product(transpose_matrix(x_matrix), x_matrix))), transpose_matrix(x_matrix)), y_matrix)
    return coefficient_matrix

def get_coefficient_std_errors(x_matrix, df, y_true, y_pred):
    
    inverted_matrix = np.linalg.inv(dot_product(transpose_matrix(x_matrix), x_matrix))
    
    diagonal_matrix = get_matrix_diagonal(inverted_matrix)
    mse = sum(((df[y_true] - df[y_pred]) ** 2))/len(df)
    std_error_matrix = np.sqrt(mse*diagonal_matrix)
    
    return std_error_matrix

# ###### Run Analysis

#Create x and y matrix
random.seed(0)
xmatrix = create_matrix(2, 2, 3, 8) 
ymatrix = create_matrix(2, 1, 3, 8) 
print(xmatrix)
print(ymatrix)

#Determinant for a two by two matrix
# determinant = (xmatrix[0][0]*xmatrix[1][1]) - ((xmatrix[0][1]*xmatrix[1][0]))
# determinant

#Find determinant and inverse using numpy
determinant = np.linalg.det(np.dot(xmatrix,np.transpose(xmatrix)))
print(f'determinant: {determinant}')

inverse_xmatrix = np.linalg.inv(np.dot(xmatrix,np.transpose(xmatrix)))
inverse_xmatrix

#Create identity matrix same length as xmatrix
identity_matrix = create_identity_matrix(len(xmatrix), len(transpose_matrix(xmatrix)[0]))
identity_matrix

#Transpose a matrix and compare my created function to numpys built in one
transposed_xmatrix = transpose_matrix(xmatrix)
print(transposed_xmatrix)
print(np.transpose(xmatrix))

#take dot product of x and transposed x matrix and compare formula output to numpy's
print(dot_product(xmatrix, transpose_matrix(xmatrix)))
np.dot(xmatrix, transpose_matrix(xmatrix))

#Use function to get matrix diagonal
diagonal_matrix = get_matrix_diagonal(xmatrix)
diagonal_matrix

#Compare function output to numpy's diagonal function
np.diag(xmatrix)

#Calculate simulated coefficients
coefficients_matrix = regression_formula(xmatrix, ymatrix)
coefficients_matrix

#Coefficient formula using numpy for comparison
np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(xmatrix),xmatrix)), np.transpose(xmatrix)), ymatrix)

# ### Exploratory Data Analysis
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

#Set chart background to seaborn and import dataset
sn.set()
data = pd.read_csv(r'C:\Users\nickp\Desktop\data science\baby weight\births.csv')
data.head()

#Dictionaries for mapping
gender = {1: 'male',
       2: 'female'}

marital_status = {1: 'married',
                 2: 'not married'}

smoker = {1: 'yes',
         0: 'no'}

#premature defined as less than 36 weeks
premie = {0: 'no',
         1: 'yes'}

#lists for mapping dictionaries to new columns
numeric_to_categorical_dicts = [gender, marital_status, smoker, premie]
mapping_columns_list = ['Sex', 'Marital', 'Smoke', 'Premie']

#Zip above lists to run mapping of dictionaries
for a, b in zip(numeric_to_categorical_dicts, mapping_columns_list):
    data[b+'_category'] = data[b].map(a)

#Check data types
data.dtypes

#check null values
data.isnull().sum()

#Check data shape
data.shape

#summary statistics
data.describe()

#Run correlation metrics
data.corr().style.background_gradient(cmap='coolwarm').set_precision(2)

#Distribution plot of y variable
sn.histplot(data['BirthWeightOz'])

#Run scatter plots to see linear relationship between variables and birth weight and save all charts as a pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("scatterplots.pdf")

scatter_columns = ['Mom Age', 'Weight Gained', 'Weeks']

for i in scatter_columns:
    plot2 = plt.figure()
    plt.figure(figsize=(10,8))
    plt.title("Effect of {} on Birth Weight".format(i))
    plt.xlabel(i)
    plt.ylabel('Birth Weight')
    plt.scatter(data[i], data['BirthWeightOz'])
    pdf.savefig(plot2)    
pdf.close()

#Show boxplots of categorical columns
cat_columns = ['Sex_category','Marital_category', 'Smoke_category']
for i in cat_columns: 
    ax = sn.boxplot(x=i, y='BirthWeightOz', data=data)
    ax.set_title(f'Effect of {i.split("_", 1)[0]} on birth rate')
    ax.set_ylabel('Birth Weight')
    ax.set_xlabel(i.split("_", 1)[0])
    plt.show()

# ##### Hypothesis Testing
from scipy import stats

#Variance formula using a dataframe and column as an input
def get_variance(df, target_column):
    total = 0
    for i in range(0, len(df)):
        squared_sum = (df[target_column].iloc[i] - df[target_column].mean())**2
        total += squared_sum
    
    variance = total/(len(df)-1)
    return variance

#Formula for calculating two sample t test by hand
def get_twosamplettest(df, target_column, ttest_dict, equal_variance=True):
    test_df = pd.DataFrame()
    for key, values in ttest_dict.items():
        if equal_variance == True:
            group_1 = pd.DataFrame(df.loc[data[key] == values[0], target_column])
            group_2 = pd.DataFrame(df.loc[data[key] == values[1], target_column])
            
            pooled_se = np.sqrt((((len(group_1)-1)*get_variance(group_1, target_column))+((len(group_2)-1)*get_variance(group_2, target_column)))/(len(group_1)+len(group_2)-2))
            t = (group_1[target_column].mean() - group_2[target_column].mean())/(pooled_se*np.sqrt(1/len(group_1)+1/len(group_2)))
            t_df = pd.DataFrame({'variable':key,
                                't stat':t}, index=[0])

            test_df = test_df.append(t_df, ignore_index=True, sort=False)
        else:
            group_1 = pd.DataFrame(df.loc[data[key] == values[0], target_column])
            group_2 = pd.DataFrame(df.loc[data[key] == values[1], target_column])

            t = (group_1[target_column].mean() - group_2[target_column].mean())/ np.sqrt((get_variance(group_1, target_column)/len(group_1)) + (get_variance(group_2, target_column)/len(group_2)))

            t_df = pd.DataFrame({'variable':key,
                                't stat':t}, index=[0])

            test_df = test_df.append(t_df, ignore_index=True, sort=False)

    return test_df

#Run any confidence interval for any dataframe column
def get_confidence_interval(df, column, ci=.95):
    col_mean = df[column].sum()/len(df[column])
    upper_bound = col_mean + stats.t.ppf(1- ((1-ci)/2), len(df)-1)*(np.sqrt(get_variance(df, column))/np.sqrt(len(df)))
    lower_bound = col_mean - stats.t.ppf(1- ((1-ci)/2), len(df)-1)*(np.sqrt(get_variance(df, column))/np.sqrt(len(df)))
    return (lower_bound, upper_bound)

#Use Scipy to compute a two sample t test on the binary categorical variables
#Create dictionary of columns and values where there must be two values in the list per category for the two sample test
categorical_dict = {'Sex_category':['male', 'female'],
                   'Marital_category':['married','not married'],
                   'Smoke_category':['no','yes'],
                   'Premie_category':['no','yes']}

for key, values in categorical_dict.items():
    group1 = pd.DataFrame(data.loc[data[key] == values[0], 'BirthWeightOz'])
    group2 = pd.DataFrame(data.loc[data[key] == values[1], 'BirthWeightOz'])
    print(f'{key}: {stats.ttest_ind(group1["BirthWeightOz"], group2["BirthWeightOz"], equal_var=True)}')

#Run function to compare output to scipy
get_twosamplettest(data, 'BirthWeightOz', categorical_dict, equal_variance=True)

#Use Scipy to calculate any confidence interval of sample birthweight mean
stats.t.interval(0.95, len(data)-1, loc=np.mean(data['BirthWeightOz']), scale=stats.sem(data['BirthWeightOz']))

#Use function to get confidence interval at any confidence level
get_confidence_interval(data, 'BirthWeightOz', ci=.95)

#Run confidence intervals on categorical data column 
for i in data['Sex_category'].unique():
    df2 = data.copy()
    df2 = df2.loc[df2['Sex_category'] == i]
    print(f' {i} confidence internval {get_confidence_interval(df2, "BirthWeightOz", ci=.95)}')
#After comparing confidence intervals, we cannot conclude there is a statistically significant 
#difference between male and female babies mean birthweights at the 95% confidence interval

# ### Regression Modelling

# ##### Model Setup
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Create a function to create the input x and y data for the model
def create_x_y(df, x_columns, y_column, variable_drop_cat_dict):
    drop_columns = [key + '_' + value for key, value in variable_drop_cat_dict.items()]
    x = df.copy()
    x = x[x_columns]
    x = pd.get_dummies(x, columns=variable_drop_cat_dict.keys())
    
    x = x.drop(drop_columns, axis=1)
    x = sm.add_constant(x)
    
    y = df.copy()
    y = y[[y_column]]
    
    return x, y

#define function to get prediction -- need to exponentiate if logged
def get_predictions(coeffs, x_population):
    x_population['Id'] = np.arange(len(x_population))
    column_list = list(x_population.columns)
    column_list.remove('Id')
    x_df = pd.melt(x_population, id_vars=['Id'], value_vars=column_list)
    x_df = pd.merge(x_df, coeffs, left_on='variable', right_on = 'Variable', how='left')
    x_df['value'].fillna(0, inplace=True)
    
    x_df['Prediction Components'] = x_df['Coefficient']*x_df['value']
    predictions_df = x_df.groupby('Id')['Prediction Components'].sum().reset_index()
    predictions_df.rename(columns={'Prediction Components':'Prediction'}, inplace=True)
    predictions_df.drop('Id', axis=1, inplace=True)
    x_population.drop('Id', axis=1, inplace=True)
    return predictions_df

#create function to calculate r2 for any linear regression model
def get_r2(df, y_true, y_pred):
    df['Squared Differences'] = (df[y_true] - df[y_pred])**2
    
    ymean = df[y_true].mean()
    df['ymean'] = ymean
    
    df['Total Variability'] = (df[y_true] - df['ymean'])**2
    
    r2 = (df['Total Variability'].sum() - df['Squared Differences'].sum())/df['Total Variability'].sum()
    return r2

#Calculate metrics for Regression Model
def regression_metrics(df, y_true, y_pred): 
    mse = sum(((df[y_true] - df[y_pred]) ** 2))/len(df)
    rmse = np.sqrt(np.mean((df[y_true] - df[y_pred]) ** 2))
    r2 = get_r2(df, y_true, y_pred)
    mae = sum(abs(df[y_true] - df[y_pred]))/len(df)
    metrics_df = pd.DataFrame({'MSE': mse,
                               'RMSE': rmse,
                               'R2':r2,
                               'MAE':mae}, index=[1])
    
    return metrics_df

#Lists for splitting data into x and y sets
x_variables = ['Sex_category', 'Marital_category', 'Smoke_category', 'Weeks', 'Weight Gained']
y_variable = 'BirthWeightOz'

cat_variable_dict_drop_categories = {'Sex_category': 'male',
                                     'Marital_category': 'not married',  
                                     'Smoke_category': 'no'}

#Run function to get x and y split
x, y = create_x_y(data, x_variables, y_variable, cat_variable_dict_drop_categories)

# ##### Run Models

#Use stats models to run regression
lr = sm.OLS(np.log(y), x)
model = lr.fit()
model.summary()

#Calculate coefficients using linear algebra formulas from part 1
coefficients = regression_formula(x.values, np.log(y.values))
coefficients

#Do formula with numpy to compare
np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x.values),x.values)), np.transpose(x.values)), np.log(y.values))

#Create a coefficient dataframe
coeff_df = pd.DataFrame(coefficients, 
             columns=['Coefficient'])

variable_df = pd.DataFrame(x.columns, 
             columns=['Variable'])

coeff_df = coeff_df.merge(variable_df, how='inner', left_index=True, right_index=True)
coeff_df

# ##### Get Predictions and evaluate performance

#Get Predictions from stats models
sm_pred = pd.DataFrame(model.predict(x))
sm_pred.head()

#run predictions using calculated coefficients and predictions function and compare to stats models output
predictions = get_predictions(coeff_df, x)
predictions.head()

#Merge predictions onto original dataframe and log the original
data = pd.merge(data, predictions, left_index=True,right_index=True, how='inner')
data['LoggedBirthWeightActual'] = np.log(data['BirthWeightOz'])
data['Exponentiated Prediction'] = np.exp(data['Prediction'])

#Calculate Regression metrics
metrics = regression_metrics(data, 'LoggedBirthWeightActual','Prediction')
metrics

#Run metrics using sklearn to compare
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

print(f'MSE: {mean_squared_error(np.log(y), sm_pred)}')
print(f'RMSE: {np.sqrt(mean_squared_error(np.log(y), sm_pred))}')
print(f'R2: {r2_score(np.log(y), sm_pred)}')
print(f'MAE: {mean_absolute_error(np.log(y), sm_pred)}')

#Get standard errors of coefficient and compare to oens from 
coeff_std_errors = pd.DataFrame(get_coefficient_std_errors(x.values, data, 'LoggedBirthWeightActual','Prediction'),
    columns=['Standard Error'])
coeff_std_errors

#Merge standard error with coefficient dataframe and calculate test statistic for statistical significance
coeff_df = pd.merge(coeff_df, coeff_std_errors, left_index=True, right_index=True, how='inner')
coeff_df['T Statistic'] = coeff_df['Coefficient'] / coeff_df['Standard Error']
coeff_df

#calculate vif scores using statsmodels
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns
vif

#Write Dataframes to excel
writer = pd.ExcelWriter(r'C:\Users\nickp\Desktop\data science\baby weight\baby weight regression output.xlsx')
coeff_df.to_excel(writer, "Coefficients", index=False)
data.to_excel(writer, "dataset with pred", index=False)
metrics.to_excel(writer, "metrics", index=False)
writer.save()