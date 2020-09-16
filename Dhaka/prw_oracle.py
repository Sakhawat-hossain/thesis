# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 13:07:09 2019

@author: hossain
"""

from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import pandas as pd
import csv
from datetime import datetime

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=18)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=10)

# Where to save the figures

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

#sp = 138
#dp = 189
locationName = ['Azimpur', 'Gabtoli', 'Uttara Sector 1', 'Mohakhali', 'Badda Bus Stop']
location = [1, 3, 4, 5, 8]
weekdays = ['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']
color = ['b','r','g','grey','m','y','k']
#b : blue. g : green. r : red. c : cyan. m : magenta. y : yellow. k : black. w : white.

def prepare_data(send_data,day,sp,dp):
    send_data = send_data[(send_data.Starting_point_id==sp) & (send_data.Dropping_point_id==dp) & (send_data.Weekday==day)]
    #selected_data.sort_values(by="pickup_time", inplace=True)
    #weekday
    
    #print('hello')
    return send_data

#import os

# Code example

# ------------------------ Load the data --------------------------- #

#selected_data = pd.read_csv("selected_data.csv", thousands=',')
selected_data = pd.read_csv("coefficient_weekday.csv", thousands=',')
input_data = pd.read_csv("input_data.csv", thousands=',')
zone = pd.read_csv("zone_lookup.csv", thousands=',')


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
    
model = LinearRegression()
deg = 8

i = 0 
j = 0 # pair of location

#-------------------------- get and show result ----------------------------#

tr = 1
y_pred = np.array([0.0])
temp = np.array([0.0])

spi = location[0] #input('Starting locationID : ') 1
dpi = location[2] #input('Destination locationID : ') 4

'''
print('Comparative travel time from ', locationName[spi], ' to ', locationName[dpi])

xtest_data=0
ytest_data=0

i = 0
while i<7: # for days
    y_pred = np.array([0.0])
    data = prepare_data(selected_data,weekdays[i],spi,dpi)
    
    #predict and test with data of next month
    
    # Prepare the data
    test_data1 = input_data #prepare_data(input_data,weekdays[i],spi,dpi)

    x_test = pd.to_datetime(test_data1.Pickup_datetime).dt.time
    #test_data1 = test_data1.sort_values(by="Pickup_time_min") #, inplace=True

    X_ = np.c_[test_data1.Pickup_time_min] 
    
    X_test = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
    
    j = 0
    k = 0
    
    while j<len(X_test):
        if j==0:
            y_pred[0] = data.Intercept
        else:
            temp[0] = data.Intercept
            y_pred = np.concatenate([y_pred,temp])
        #
        k = 0
        while k<deg:
            y_pred[j] = y_pred[j] + X_test[j][k]*data['Power_'+str(k)]
            k = k+1
        j = j+1             
    
    plt.plot(x_test,y_pred,color[i],label=weekdays[i])
    plt.locator_params(nbins=6,axis='x')
 
    i = i+1
plt.xlabel('Pickup time')
plt.ylabel('Duration (minutes)')
plt.legend()
plt.show()
   


cur = datetime.now()
day = cur.strftime("%A")
print('Travel time today (', day, ') from ', locationName[spi], ' to ', locationName[dpi])

i = 0
while i<1: # for days
    y_pred = np.array([0.0])    
    data = prepare_data(selected_data, day, spi, dpi)
    
    # Prepare the data
    test_data1 = input_data #prepare_data(input_data,weekdays[i],spi,dpi)

    x_test = pd.to_datetime(test_data1.Pickup_datetime).dt.time
    #test_data1 = test_data1.sort_values(by="Pickup_time_min") #, inplace=True

    X_ = np.c_[test_data1.Pickup_time_min] 
    
    X_test = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
    
    j = 0
    k = 0
    
    while j<len(X_test):
        if j==0:
            y_pred[0] = data.Intercept
        else:
            temp[0] = data.Intercept
            y_pred = np.concatenate([y_pred,temp])
        #
        k = 0
        while k<deg:
            y_pred[j] = y_pred[j] + X_test[j][k]*data['Power_'+str(k)]
            k = k+1
        j = j+1             
    
    plt.plot(x_test,y_pred,color[i],label=day)
    plt.locator_params(nbins=6,axis='x')
 
    i = i+1
plt.xlabel('Pickup time')
plt.ylabel('Duration (minutes)')
#plt.legend()
plt.show()
'''
# -------------- Least congested path ----------------------- #
num_route = 3
route = [[1, 3, 4], [1, 5, 4], [1, 8, 5, 4]]
#routeName = np.array(len(route))

#cur = datetime.now()
cur = datetime(2020, 9, 1, 10, 0, 0)

day = cur.strftime("%A")
print('Least Congested Path from ', locationName[spi], ' to ', locationName[dpi], ' (', day, ')')
print()
print('starting time  :  ', cur)

min_ttime = 999999
idx = 0
X_ = np.array([[0.0]])
i = 0
while i<num_route: # for every route   
    X_[0][0] = cur.hour*60 + cur.minute
    y_total = 0.0
    
    j = 1
    while j < len(route[i]): # for each pair
        spi = route[i][j-1]
        dpi = route[i][j]
        data = prepare_data(selected_data, day, spi, dpi)
        
        X_coeff = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
        
        y_pred = data.Intercept
        
        k = 0
        while k<deg:
            y_pred = y_pred + X_coeff[0][k]*data['Power_'+str(k)]
            k = k+1            
        j = j+1
        X_[0][0] = X_[0][0] + y_pred
        y_total = y_total + y_pred.values[0]
    if min_ttime > y_total:
        min_ttime = y_total
        idx = i
    routeName = zone[zone.LocationID == route[i][0]].Zone.values[0]
    k=1
    while k<len(route[i]):
        routeName = routeName + " -> " + zone[zone.LocationID == route[i][k]].Zone.values[0]
        k = k+1
    routeName = routeName + "   =>   "
    print(routeName, str(y_total))
    
    i = i+1


