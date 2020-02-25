# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:59:56 2019

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

#sp = 79
#dp = 146
sp = 138
dp = 189

def prepare_data(send_data):
    send_data = send_data[(send_data.starting_point_id==sp) & (send_data.dropping_point_id==dp)]
    #selected_data.sort_values(by="pickup_time", inplace=True)
    
    print('hello')
    return send_data

#import os

# Code example

# Load the data
selected_data = pd.read_csv("selected_data_5.csv", thousands=',')

# Prepare the data
data = prepare_data(selected_data)

#select data with specified day
#date = "2019-01-10"
#data = data[data["pickup_time"].str.contains(date)]
'''
date = "2019-01-30" #"2019-01-10"
data1 = data[data["pickup_time"].str.contains(date)]

date = "2019-01-23"
data2 = data[data["pickup_time"].str.contains(date)]

date = "2019-01-16"
data3 = data[data["pickup_time"].str.contains(date)]

date = "2019-01-09"
data4 = data[data["pickup_time"].str.contains(date)]

date = "2019-01-02"
data5 = data[data["pickup_time"].str.contains(date)]
'''

#print(data)
#print(data)

data["pickup_time"] = pd.to_datetime(data["pickup_time"]).dt.time
data.sort_values(by="pickup_time_sec", inplace=True)
x_train = data["pickup_time"] #dt.date for date
y_train = data["duration"]
X_train = np.c_[data["pickup_time_sec"]] 
#X = mpl.dates.date2num(X)

#data.plot(kind='line', x="pickup_time", y='duration')
plt.plot(x_train,y_train)
plt.locator_params(nbins=6,axis='x')
plt.show()

# Visualize the data
# Select a linear model

import sklearn.linear_model 
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X_train, y_train)

#predict and test with data of next month
test_data = pd.read_csv("test_data_5.csv", thousands=',')

# Prepare the data
test_data = prepare_data(test_data)

test_data["pickup_time"] = pd.to_datetime(test_data["pickup_time"]).dt.time
test_data.sort_values(by="pickup_time_sec", inplace=True)
x_test = test_data["pickup_time"] #dt.date for date
y_test = test_data["duration"]
X_test = np.c_[test_data["pickup_time_sec"]]

y_pred = model.predict(X_test)

plt.plot(X_test,y_pred,'g')
plt.scatter(X_test,y_test)
plt.locator_params(nbins=6,axis='x')
plt.show()

from sklearn import metrics
print('MAE : ',metrics.mean_absolute_error(y_test, y_pred))
print('MSE : ',metrics.mean_squared_error(y_test, y_pred))
print('RMSE : ',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

'''
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%d')
ax.xaxis.set_major_formatter(myFmt)


data.plot(kind='line', x="pickup_time", y='duration')
plt.show()


# Visualize the data
# Select a linear model
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, Y)

# Make a prediction for Cyprus
X_new = [['1/1/2019 1:01']]  # Cyprus' GDP per capita 22587
print(model.predict(X_new)) # outputs [[ 5.96242338]]
'''