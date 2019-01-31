# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model


"---------- ANÁLISE PREDITIVA ----------"

#Definition of dependent and independent variable and data data division

def DefineXandY(dataframe):
    # Define variable x and y
    variable_x = dataframe.iloc[:,25:].values
    variable_y = dataframe['standard_value'].values
    return variable_x,variable_y


def Holdout(x,y):
    # Division of data in train and test (Holdout)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    return x_train,y_train,x_test,y_test


#Application of ML algorithms

def SVRPrediction(x_train,y_train, x_test):
    #Prevision with SVR using different kernel parameters and C and gamma value constants
    #Return a dictionary with the predictions of each different kernel parameters
    dictionary = {}
    svr_kernel = ['poly', 'rbf', 'sigmoid', 'linear']
    for k in svr_kernel:
        model_SVR = SVR(kernel=k, C=1e3, gamma=0.1)
        model_SVR.fit(x_train, y_train)
        pred_SVR = model_SVR.predict(x_test)
        dictionary[k] = pred_SVR
    return dictionary


def RandomForestPrediction(x_train,y_train, x_test):
    #Prevision with Random Forest Regression using different values for number of estimators
    #Return a dictionary with the predictions of each different number of estimators
    dictionary = {}
    n_estimators = [10,50,100,500,1000]
    for estimator in n_estimators:
        model_RFR = RandomForestRegressor(n_estimators=estimator)
        model_RFR.fit(x_train,y_train)
        pred_RFR = model_RFR.predict(x_test)
        dictionary[estimator] = pred_RFR
    return dictionary

def LinearRegression(x_train,y_train, x_test):
    #Prevision with linear regression
    #Return a dictionary with the prediction
    dictionary = {}
    model_lregr = linear_model.LinearRegression()
    model_lregr.fit(x_train,y_train)
    pred_lregr = model_lregr.predict(x_test)
    dictionary['linear'] = pred_lregr
    return dictionary


#Functions for error calculation

def CalculationError(dictionary,y_test):
    #Calculation of the error metrics of each prediction
    #Return a dictionary with the error metrics (MAE,MSE,RMSE) associated to the each parameters
    dictionary_error = {}
    for key, value in dictionary.items():
        mean_abs_error = metrics.mean_absolute_error(y_test, value)
        mean_sq_error = metrics.mean_squared_error(y_test, value)
        root_mean_sq_error = math.sqrt(metrics.mean_squared_error(y_test, value))
        dictionary_error[key] = {'MAE': str(mean_abs_error),
                                 'MSE': str(mean_sq_error),
                                 'RMSE': str(root_mean_sq_error)
                                 }
    return dictionary_error

def BestErrorParameters(dictionary_error):
    #Select the best parameters based in the minimum value of MAE, MSE, RMSE
    #Return a tuple with the best parameters
    list_mae = []
    list_mse = []
    list_rmse = []
    for key, value in dictionary_error.items():
        mae_values = (key, float(value['MAE']))
        mse_values = (key, float(value['MSE']))
        rmse_values = (key, float(value['RMSE']))
        list_mae.append(mae_values)
        list_mse.append(mse_values)
        list_rmse.append(rmse_values)
    ordenation_mae = sorted(list_mae, key=lambda tup: tup[1], reverse=False)
    ordenation_mse = sorted(list_mse, key=lambda tup: tup[1], reverse=False)
    ordenation_rmse = sorted(list_rmse, key=lambda tup: tup[1], reverse=False)
    return ordenation_mae[0],ordenation_mse[0],ordenation_rmse[0]

def AccuracyModel(mae_value):
    #Calculate the accuracy of each machine learning model using MAE value
    accuracy = 100-np.mean(mae_value)
    return accuracy



#Retrive the predictions of each machine learning model

def RetrivePredictions(dictionary_error_svr, dictionary_error_rf,
                       dictionary_error_lr, model_svr, model_rf,
                       model_lr):
    #Retrive the machine learning predictions with the best hyper parameters
    #Return a list of the predictions
    best_parameter_svr = BestErrorParameters(dictionary_error_svr)[0][0]
    best_parameter_rf = BestErrorParameters(dictionary_error_rf)[0][0]
    best_parameter_lr = BestErrorParameters(dictionary_error_lr)[0][0]
    prevision_svr = list(model_svr[best_parameter_svr])
    prevision_rf = list(model_rf[best_parameter_rf])
    prevision_lr = list(model_lr[best_parameter_lr])
    return prevision_svr, prevision_rf, prevision_lr



def AddPredictionDataFrame(file_name, pred_svr, pred_rf, pred_lr):
    #Add the predictions of the machine learning into the csv file format
    dataframe_svr  = pd.read_csv(file_name)
    dataframe_rf = pd.read_csv(file_name)
    dataframe_lr = pd.read_csv(file_name)
    dataframe_svr['pKd_[M]_pred'] = pred_svr
    dataframe_rf['pKd_[M]_pred'] = pred_rf
    dataframe_lr['pKd_[M]_pred'] = pred_lr
    return dataframe_svr.to_csv('lr_predictions.csv'),\
           dataframe_rf.to_csv('rf_predictions.csv'),\
           dataframe_lr.to_csv('svr_predictions.csv')



if __name__=="__main__":
    #Import dataset training and dataset test
    dataset_training = 'final_dataset.csv'
    dataset_test = 'round_1_template.csv'
    df_train = pd.read_csv(dataset_training)
    df_test = pd.read_csv(dataset_test)

    #Definition of x and y variable
    x_df_train,y_df_train = DefineXandY(df_train)
    x_df_test = df_test

    #Division of data in train data and test data
    data_division = Holdout(x_df_train,y_df_train)
    
###############################################################
    
" ---------- Linear Regression Algorithm Implementation ---------- "

    lr_prediction = LinearRegression(data_division[0], data_division[1], x_df_test)
    lr_error  =CalculationError(lr_prediction, data_division[3])
    lr_best_parameters = BestErrorParameters(lr_error)
    accuracy_lr = AccuracyModel(lr_best_parameters[0][1])
    print('Linear Regression:')
    print('MAE value -->', round(lr_best_parameters[0][1], 2))
    print('MSE value -->', round(lr_best_parameters[1][1], 2))
    print('RMSE value -->', round(lr_best_parameters[2][1], 2))
    print('Accuracy with the best parameters -->', round(accuracy_lr, 2))
    
###############################################################

" ---------- Random Forest Regression Algorithm Implementation ---------- "

    rf_prediction = RandomForestPrediction(data_division[0],data_division[1],x_df_test)
    rf_error = CalculationError(rf_prediction,data_division[3])
    rf_best_parameters = BestErrorParameters(rf_error)
    accuracy_rf = AccuracyModel(rf_best_parameters[0][1])
    print('Random Forest Regression:')
    print('MAE value -->', round(rf_best_parameters[0][1], 2))
    print('MSE value -->', round(rf_best_parameters[1][1], 2))
    print('RMSE value -->', round(rf_best_parameters[2][1], 2))
    print('Accuracy with the best parameters -->', round(accuracy_rf, 2))
    
###############################################################

" ---------- Support Vector Machine Regression Algorithm Implementation ---------- "

    svr_prediction = SVRPrediction(data_division[0],data_division[1],x_df_test)
    svr_error = CalculationError(svr_prediction,data_division[3])
    svr_best_parameters = BestErrorParameters(svr_error)
    accuracy_svr =AccuracyModel(svr_best_parameters[0][1])
    print('Support Vector Machine Regression:')
    print('MAE value -->', round(svr_best_parameters[0][1], 2))
    print('MSE value -->', round(svr_best_parameters[1][1], 2))
    print('RMSE value -->', round(svr_best_parameters[2][1], 2))
    print('Accuracy with the best parameters -->', round(accuracy_svr, 2))
    
###############################################################

    #Retrive information of predictions
    predictions = RetrivePredictions(dictionary_error_svr=svr_error,
                             dictionary_error_rf=rf_error,
                              dictionary_error_lr=lr_error,
                              model_svr=svr_prediction,
                              model_rf=rf_prediction,
                              model_lr=lr_prediction)
    pred_svr = predictions[0] #list of predictions SVR
    pred_rf = predictions[1]  #list of predictions Random Forest
    pred_lr = predictions[2]  #list of predictions Linear Regression
    print('Prediction using SVR',pred_svr)
    print('Prediction using Random Forest',pred_rf)
    print('Prediction using Linear Regression',pred_lr)

    #Information about predictions in csv file
    file_name = 'round_1_predictions.csv'
    add_prediction = AddPredictionDataFrame(file_name, pred_svr, pred_rf,pred_lr)
    print('Files in format csv with the predictions of each machine learning algorithm are available!')