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


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
    
model = LinearRegression()
deg = 8

i = 0 
j = 0 # pair of location
num_loc = 5 #number of locations
dloc = 0
sloc = 0
'''
with open('coefficient_weekday.csv', mode='w', newline='') as file:
    # file contains expected data having following field
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
    writer.writerow(['Starting_point_id','Dropping_point_id','Weekday','Intercept','Power_0',
                     'Power_1','Power_2','Power_3','Power_4','Power_5','Power_6','Power_7'])


with open('coefficient_weekday.csv', mode='a', newline='') as file:
    # file contains expected data having following field
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) 
    while i<num_loc:
        sloc = location[i]
        j = 0
        while j<num_loc:
            if i==j:
                j = j+1
                continue
            k = 0
            dloc = location[j]
            while k<7: # for 7 days
                
                data = prepare_data(selected_data,weekdays[k],sloc,dloc)
                
                if data.empty:
                    print('empty')
                    k = k+1
                    continue
                
                #data.Pickup_datetime = pd.to_datetime(data.Pickup_datetime).dt.time
                
                data = data.sort_values(by=['Pickup_time_min']) #, inplace=True
                #data.Pickup_time_min = data.Pickup_time_min + i*1440
                   
                #x_train = data.Pickup_datetime #dt.date for date
                y_train = data.Duration
                X = np.c_[data.Pickup_time_min] 
                #X = mpl.dates.date2num(X)
                    
                X_train = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X) 
                        # Train the model
                model.fit(X_train, y_train)
                
                intercept = model.intercept_
                coef = model.coef_
                
                writer.writerow([sloc,dloc,weekdays[k],intercept,coef[0],coef[1],coef[2],coef[3],
                                 coef[4],coef[5],coef[6],coef[7] ])
                
                k = k+1
            j = j+1
        i = i+1
    #flag = input('Enter letter to run or press c to cancel : ')
    #if flag == 'c':
    #break
'''

#-------------------------- get and show result ----------------------------#

tr = 1
y_pred = np.array([0.0])
temp = np.array([0.0])
while tr<2:
    
    spi = location[0] #input('Starting locationID : ')
    dpi = location[2] #input('Destination locationID : ')
    
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
                
        #x_test = test_data1.Pickup_datetime #dt.date for date
        #y_test = test_data1.Duration
        X_ = np.c_[test_data1.Pickup_time_min]
        
        #test_x = test_data1.pickup_time_min + i*1440
        
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
        
        #print(len(X_test))
        #print(y_pred[0])
        plt.plot(x_test,y_pred,color[i],label=weekdays[i])
        plt.locator_params(nbins=6,axis='x')
       
        #plt.plot(deg_arr,rmse_arr,color[i]
        
        #print('MAE : ',metrics.mean_absolute_error(y_test, y_pred))
        #print('MSE : ',metrics.mean_squared_error(y_test, y_pred))
        #print('RMSE : ',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        #print(' ')
        
        i = i+1
    plt.xlabel('Pickup time')
    plt.ylabel('Duration (minutes)')
    plt.legend()
    plt.show()
   
    #flag = input('Enter letter to run or press c to cancel : ')
    #if flag == 'c':
    #break
    tr = tr+1
