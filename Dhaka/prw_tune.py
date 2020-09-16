# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 13:07:09 2019

@author: hossain
"""

from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import pandas as pd

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=8)

# Where to save the figures

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

def prepare_data(send_data,day,sp,dp):
    send_data = send_data[(send_data.Starting_point_id==sp) & (send_data.Dropping_point_id==dp) & (send_data.Weekday==day)]
    #selected_data.sort_values(by="pickup_time", inplace=True)
    #weekday
    
    return send_data

#import os

# Code example

# Load the data
selected_data = pd.read_csv("selected_data.csv", thousands=',')

# Prepare the data
#select data with specified day

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

weekdays = ['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']
color = ['b','r','g','grey','m','y','k']

#b : blue. g : green. r : red. c : cyan. m : magenta. y : yellow. k : black. w : white.

i = 0
while i<7:
    
    data = prepare_data(selected_data,weekdays[i])
    
    data.pickup_time = pd.to_datetime(data.pickup_time).dt.time
    
    data = data.sort_values(by=['pickup_time_min']) #, inplace=True
    x_train = data.pickup_time #dt.date for date
    y_train = data.duration
    X = np.c_[data.pickup_time_min] 
    #X = mpl.dates.date2num(X)

    #data.plot(kind='line', x="pickup_time", y='duration')
    #plt.plot(x_train,y_train)
    #plt.locator_params(nbins=6,axis='x')
    #plt.show()

    # Visualize the data
    # Select a linear model

    model = LinearRegression()
    #transformer = PolynomialFeatures(degree=2, include_bias=False)
    #transformer.fit(X)
    #X_train = transformer.transform(X)
    
    
    #predict and test with data of next month
    test_data = pd.read_csv("test_data_weekday.csv", thousands=',')
    
    # Prepare the data
    test_data = prepare_data(test_data,weekdays[i])

    # For certain day
    #date = "2019-04-10" 
    #test_data = test_data[test_data["pickup_time"].str.contains(date)]

    test_data.pickup_time = pd.to_datetime(test_data.pickup_time).dt.time
    test_data = test_data.sort_values(by="pickup_time_min") #, inplace=True
    x_test = test_data.pickup_time #dt.date for date
    y_test = test_data.duration
    X_ = np.c_[test_data.pickup_time_min]
    
    itune = 0
    deg = 2
    rmse_min = 9999999
    deg_arr = []
    rmse_arr = []
    
    while itune<15:
    
        X_train = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X) 
        # Train the model
        model.fit(X_train, y_train)
        
    
        X_test = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        
        print(rmse)
        deg_arr.append(deg)
        rmse_arr.append(rmse)
        
        itune = itune+1
        deg = deg+1

    plt.plot(deg_arr,rmse_arr,color[i],label=weekdays[i])
    #plt.plot(x_test,y_pred,color[i])
    #plt.plot(x_test,y_test)
    plt.locator_params(nbins=6,axis='x')
    #plt.show()

    #print('MAE : ',metrics.mean_absolute_error(y_test, y_pred))
    #print('MSE : ',metrics.mean_squared_error(y_test, y_pred))
    #print('RMSE : ',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(' ')
    
    i = i+1

plt.legend()
plt.show()
