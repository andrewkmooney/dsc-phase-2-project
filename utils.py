import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats as stats

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Residual plots from Statsmodels model
def regress_plots(columns, model):
    
    '''
    Takes in a list of column names found in the DataFrame that is used in a statsmodel linear regression and 
    plots 4 Regression Plots for each variable to show residuals per variable.
    
    Parameters
    ----------
    columns:    (list) - List of column names belonging to DataFrame used in statsmodel linear regression model
    model:    (sm model) - a statsmodel linear regression model that contains the list of columns.
    
    Returns
    ----------
    A regress plot of all residuals for each column

    '''    
    for column in columns:
        fig = plt.figure(figsize=(15,8))
        fig = sm.graphics.plot_regress_exog(model, column, fig=fig)
        plt.show()

def qqplot(model):
    '''
    Takes in a StatsModel linear regression and plots a residual QQ Plot to check for normality and homoscedasticity.

    Parameters
    -------
    model:    (sm model) - a StatsModel fitted linear regression model

    Outputs
    --------
    A QQ Plot of the residuals to check for normality
    
    '''
    residuals = model.resid
    fig = sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)

def log_transform(df, column):
    
    '''
    Takes in a column name from the main data dataframe and creates 2 subplots showing histograms to compare 
    the differences between the original data and the log of that data.
    
    Parameters
    -----------
    df:    (DataFrame) - A DataFrame containing the columns being investigated
    column:    (str) - the column name of the data to be transformed
    
    Outputs
    ------------
    2 side-by-side subplots showing histograms of the original data and the log of that data
    '''
    
    plt.figure(figsize=(10,5))
    ax1 = plt.subplot(1, 2, 1)
    df[column].plot.hist(ax=ax1, edgecolor='black')
    ax1.set_title(f'{column}')
    
    column_log = np.log1p(abs(df[column]))
    ax2 = plt.subplot(1, 2, 2)
    plt.hist(column_log, edgecolor='black')
    ax2.set_title(f'Log of {column}')

def sk_linear_regression(df, predictors, outcome, log=False, scaled=False, random_seed=1066):
    
    '''
    Creates a linear regression model in Sci Kit Learn using a dataframe and defined predictors/outcomes 
    using a random split of train/test data. Analyzes the model and provides an R2 score, RMSE and MAE of the test
    and training data for comparison.

    Parameters
    -----------
    df:    (DataFrame) - a DataFrame containing test data for the regression
    predictors:    (list) - a list of columns of variables to be included in 
    outcome:    (str) - the string name of y variable column for the linear regression
    random_seed:    (int) - the value of the random seed used for the train/test split. Default = 1066
    log:    (bool) - Boolean determining if the outcome should be log transformed
    scaled:    (bool) - Boolean determining if the outcome should be scaled

    Returns
    -----------
    lr:    (LinearRegression()) - A Sci-Kit Learn linear regression model
    Metrics:   Prints a report of the Root Mean Squared Error, R2 Score and Absolute Mean Error for both test and train data
    Plot:    a plot of the residuals of the test and the train data for comparison next to a histogram of both data sets
    '''
    
    lr = LinearRegression()
    
    X = df[predictors]
    y = df[outcome]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed)
    
    if log == False:
        if scaled == False:
            
            y_train_final = y_train
            y_test_final = y_test
            
            lr.fit(X_train, y_train_final)

            y_train_pred = lr.predict(X_train)
            y_test_pred = lr.predict(X_test)
            
            y_train_pred_final = y_train_pred
            y_test_pred_final = y_test_pred
        
        else:
            scaler = MinMaxScaler()
            
            y_sc = scaler.fit(np.array(y).reshape(-1.1))
            
            y_train_final = scaler.transform(np.array(y_train).reshape(-1,1))
            y_test_final = scaler.transform(np.array(y_test).reshape(-1,1))
            
            lr.fit(X_train, y_train_final)
            
            y_train_pred = lr.predict(X_train)
            y_test_pred = lr.predict(X_test)
            
            y_train_pred_final = scaler.inverse_transform(y_train_pred)
            y_test_pred_final = scaler.inverse_transform(y_test_pred)
    
    else:
        if scaled == False:
            y_train_final = np.log(y_train)
            y_test_final = np.log(y_test)

            lr.fit(X_train, y_train_final)

            y_train_pred = lr.predict(X_train)
            y_test_pred = lr.predict(X_test)

            y_train_pred_final = np.exp(y_train_pred)
            y_test_pred_final = np.exp(y_test_pred)

        else:
            scaler = MinMaxScaler()
            
            y = np.log(y)
            y_sc = scaler.fit(np.array(y).reshape(-1, 1))
            
            y_train_log = np.log(y_train)
            y_test_log = np.log(y_test)
            
            y_train_final = scaler.transform(np.array(y_train_log).reshape(-1,1))
            y_test_final = scaler.transform(np.array(y_test_log).reshape(-1,1))
            
            lr.fit(X_train, y_train_final)
            
            y_train_pred = lr.predict(X_train)
            y_test_pred = lr.predict(X_test)
            
            y_train_pred_final = np.exp(scaler.inverse_transform(y_train_pred))
            y_test_pred_final = np.exp(scaler.inverse_transform(y_test_pred))
            
    print("Training Scores:")
    print(f"R2: {r2_score(y_train_final, y_train_pred)}")

    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_train, y_train_pred_final))}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_train_pred_final)}")
    print("---")
    print("Testing Scores:")
    print(f"R2: {r2_score(y_test_final, y_test_pred)}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_test_pred_final))}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_test_pred_final)}")
            
    residuals_train = np.array(y_train_final) - np.array(y_train_pred)
    residuals_test = np.array(y_test_final) - np.array(y_test_pred)
    
    plt.figure(figsize=(15,5))
    
    ax1 = plt.subplot(1,2,1)
    
    plt.scatter(y_train_pred, residuals_train, alpha=.75)
    plt.scatter(y_test_pred, residuals_test, color='g', alpha=.75)

    plt.axhline(y=0, color='black')

    ax1.set_title('Residuals for Linear Regression Model')
    ax1.set_ylabel('Residuals')
    ax1.set_xlabel('Predicted Values')
    
    ax2 = plt.subplot(1,2,2)
    
    plt.hist(residuals_train, bins='auto', alpha=.75, edgecolor='b', label='Train')
    plt.hist(residuals_test, bins='auto', color='g', alpha=.75, edgecolor='b', label='Test')
    
    ax2.set_title('Histogram of Residuals')
    ax2.legend()
    
    plt.show()

    return lr

def sm_linear_regression(df, predictors, outcome, log = False, scaled = False, random_seed = 1066):

    '''
    Creates a linear regression model in StatsModels using a dataframe and defined predictors/outcomes 
    using a random split of train/test data. Returns a model ready for summary.

    Parameters
    -----------
    df:    (DataFrame) - a DataFrame containing test data for the regression
    predictors:    (list) - a list of columns of variables to be included in 
    outcome:    (str) - the string name of y variable column for the linear regression
    random_seed:    (int) - the value of the random seed used for the train/test split. Default = 1066

    Returns
    -----------
    model:    (OLS) - A StatsModel linear regression model with a constant added
    '''
    
    X = df[predictors]
    y = df[outcome]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1066)
    
    if log == True:
        y_train = np.log(y_train)
    else:
        pass
    
    if scaled == True:
        scaler = MinMaxScaler()
        y_train = scaler.fit_transform(np.array(y_train).reshape(-1,1))
    else:
        pass
    
    predictors_int = sm.add_constant(X_train)
    model = sm.OLS(y_train, predictors_int).fit()
    
    return model