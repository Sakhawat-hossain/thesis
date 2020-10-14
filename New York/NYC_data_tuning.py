# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 23:21:48 2020

@author: hossain
"""

from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from datetime import timedelta

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import train_test_split

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# ----------------------- Function section ------------------------- #
def prepare_data(data, sl, dl, day):
    # sl - starting locationID
    # dl - destination locationID
    data = data[(data.Starting_locationID==sl) & (data.Dropping_locationID==dl) & (data.Weekday==day)]
    data1 = data[['Pickup_datetime', 'Pickup_time_min', 'Duration']]
    
    return data1

def prepare_data_chunk(sl, chunksize, num_location, nloc):
    #data loaded into chunk
    tv_d = pd.read_csv("train_validate_dataset.csv", thousands=',', dtype={"Starting_locationID":int, "Dropping_locationID":int,
        "Pickup_datetime":str, "Dropoff_datetime":str,"Pickup_time_min":int,"Distance":float,"Duration":float,"Weekday":str}, 
        chunksize=chunksize)
    ts_d = pd.read_csv("test_dataset.csv", thousands=',', dtype={"Starting_locationID":int, "Dropping_locationID":int,
        "Pickup_datetime":str, "Dropoff_datetime":str,"Pickup_time_min":int,"Distance":float,"Duration":float,"Weekday":str},
        chunksize=chunksize)
    
    #filtered data stored into list
    tv_chunk = []
    ts_chunk = []
    for i in range (0, nloc):
        tv_chunk.append([])
        ts_chunk.append([])
        for j in range (0, num_location+1):
            tv_chunk[i].append([])
            ts_chunk[i].append([])
    
    n = 0
    for tv_data in tv_d:
        for i in range (0, nloc):
            dl = 1
            sloc = sl + i
            n += 1
            print(n)
            while dl<(num_location+1):
                data = tv_data[(tv_data.Starting_locationID==sloc) & (tv_data.Dropping_locationID==dl)]
                #data = data[['Pickup_datetime', 'Pickup_time_min', 'Duration', 'Weekday']]
                tv_chunk[i][dl].append(data)
                dl += 1
    
    n = 0
    for ts_data in ts_d:
        for i in range (0, nloc):
            sloc = sl + i
            dl = 1
            n += 1
            print(n)
            while dl<(num_location+1):
                data = ts_data[(ts_data.Starting_locationID==sloc) & (ts_data.Dropping_locationID==dl)]
                #data = data[['Pickup_datetime', 'Pickup_time_min', 'Duration', 'Weekday']]
                ts_chunk[i][dl].append(data)
                dl += 1
    
    return tv_chunk, ts_chunk

def split_train_test(data):
    train_vald_data = data[data.Pickup_datetime.str.contains('2019-01') | data.Pickup_datetime.str.contains('2019-03')
    | data.Pickup_datetime.str.contains('2019-04') | data.Pickup_datetime.str.contains('2019-05')]
    test_data = data[data.Pickup_datetime.str.contains('2019-06')]
    
    return train_vald_data, test_data
    
# -------------------- Function section end ------------------------- #

# ----------------------- Define section ------------------------- #
#without chunksize  
#train_validate_data = pd.read_csv("train_validate_dataset.csv", thousands=',', dtype={"Starting_locationID":int, "Dropping_locationID":int,
#    "Pickup_datetime":str, "Dropoff_datetime":str,"Pickup_time_min":int,"Distance":float,"Duration":float,"Weekday":str})#chunksize=10000)

#test_data = pd.read_csv("test_dataset.csv", thousands=',', dtype={"Starting_locationID":int, "Dropping_locationID":int,
#   "Pickup_datetime":str, "Dropoff_datetime":str,"Pickup_time_min":int,"Distance":float,"Duration":float,"Weekday":str})


weekdays = ['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']
color = ['b','r','g','grey','m','y','k']
#b : blue. g : green. r : red. c : cyan. m : magenta. y : yellow. k : black. w : white.

deg = 1
max_deg = 40
INFINITY = 999999
num_location = 263 #total locations
chunksize = 1000000 # chunksize

i = 0
j = 0 #pair of locations
locationID = [x for x in range(1,num_location+1)]

model = LinearRegression()
sp = 12#138
dp1 = 13#163
dp2 = 189

# -------------------------- Define section End --------------------------------#

   
# -------------------------- Training section --------------------------------#
#create_coefficient_file(max_deg)

# for 1-10     13,113
tv_df = pd.read_csv("fitting_138_train.csv", thousands=',')
ts_df = pd.read_csv("fitting_138_test.csv", thousands=',')
sl = 138
#tv_chunk, ts_chunk = prepare_data_chunk(sl, chunksize, num_location, 1)
i = sl
while i < num_location:
    if i!=sp:
        i += 1
        continue
    j = 1
    while j < num_location:
        if (j!=dp1):# & (j!=dp2):
            j += 1
            continue
        
        print(i,j)
        
        k = 3
        while k < 4: # for each day
            #prepare data for epecific pair of locations and weekday
            tv_data = prepare_data(tv_df, i, j, weekdays[k])
            ts_data = prepare_data(ts_df, i, j, weekdays[k])
            
            print('tv data',len(tv_data))
            print('ts data',len(ts_data))
            tv_data = pd.concat([tv_data, ts_data])
            print('tv data1',len(tv_data))
            print('ts data1',len(ts_data))
            tv_data, ts_data = train_test_split(tv_data, test_size=0.2)
            
            print('tv data2',len(tv_data))
            print('ts data2',len(ts_data))
            
            #tv_data = tv_data[(tv_data.Pickup_time_min > 300) & (tv_data.Pickup_time_min < 1400)]
            #ts_data = ts_data[(ts_data.Pickup_time_min > 300) & (ts_data.Pickup_time_min < 1400)]
            
            #split dataset into training and validation subsets
            train_data, validate_data = train_test_split(tv_data, test_size=0.2)
            
            train_data.sort_values(by="Pickup_time_min", inplace=True)
            x_train = pd.to_datetime(train_data.Pickup_datetime).dt.time #taking only time
            y_train = train_data.Duration
            X_tr = np.c_[train_data.Pickup_time_min]
            
            #x_validate = validate_data.Pickup_datetime #dt.date for date
            y_validate = validate_data.Duration
            X_v = np.c_[validate_data.Pickup_time_min]
            
            ts_data.sort_values(by="Pickup_time_min", inplace=True)
            x_test = pd.to_datetime(ts_data.Pickup_datetime).dt.time #taking only time
            y_test = ts_data.Duration
            X_ts = np.c_[ts_data.Pickup_time_min]
                
            #training and tuning the model; tried to find best fit
            deg = 1
            mserr_list = []
            bias_list = []
            com_list = []
            deg_list = [d for d in range(1, max_deg+1)]
            min_error = INFINITY
            min_error1 = INFINITY
            expected_deg = 1
            expected_deg1 = 1
            while deg <= max_deg:
                # y = a_0 + a_1*x + a_2*x^2 + . . . . . + a_deg*x^deg
                # transform indenpended variable x to x, x^2, x^3, ..... , x^deg
                X_train = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_tr) 
                model.fit(X_train, y_train) # Train the model
                y_pred = model.predict(X_train)
                biasval = metrics.mean_absolute_error(y_train, y_pred)
                bias_list.append(biasval)
                
                X_validate = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_v)
                y_pred = model.predict(X_validate)

                mserr = metrics.mean_squared_error(y_validate, y_pred)
                mserr_list.append(mserr)
                
                com_list.append(mserr + biasval)
                if mserr <= min_error:
                    min_error = mserr
                    expected_deg = deg
                if (mserr + biasval) <= min_error1:
                    min_error1 = mserr + biasval
                    expected_deg1 = deg
                deg += 1
                    
            '''
            plt.plot(deg_list,mserr_list,'r')
            plt.xlabel('Degree')
            plt.ylabel('Mean Square Error')
            plt.locator_params(nbins=6,axis='x')
            plt.show()
            plt.plot(deg_list,bias_list,'g')
            plt.locator_params(nbins=6,axis='x')
            #plt.show()
            plt.plot(deg_list,com_list,'k')
            plt.locator_params(nbins=6,axis='x')
            #plt.show()
            '''
            if expected_deg < 4:
                print('deg < 4')
                #continue
            #training the model with expected degree 
            X_train = PolynomialFeatures(degree=expected_deg, include_bias=False).fit_transform(X_tr) 
            model.fit(X_train, y_train) # Train the model
            #Testing the model with testing dataset, show rmse
            X_test = PolynomialFeatures(degree=expected_deg, include_bias=False).fit_transform(X_ts)
            y_pred = model.predict(X_test)
            mserr = metrics.mean_squared_error(y_test, y_pred)
            
            # plot graph with test data
            print('x train',len(x_train))
            print('x test',len(x_test))
            #plt.plot(x_train,y_train,'b')
            plt.plot(x_test,y_test,'b')
            plt.plot(x_test,y_pred,'k')
            plt.locator_params(nbins=6,axis='x')
            plt.show()
            
            #saving coefficient of the model with other parameter          
            with open('NY_yellow_taxi_model_coefficient.csv', mode='a', newline='') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                row = [i, j, weekdays[k], expected_deg, model.intercept_]
                coef = model.coef_
                t = 0
                while t < expected_deg:
                    row.append(coef[t])
                    t += 1
                #writer.writerow(row)
            
            print(j,' Expected degree :', expected_deg,'  ','RMSE :',mserr_list[expected_deg-1])
            print(' Expected degree :', expected_deg1)

            k += 1
        
        j += 1
    i += 1
# -------------------------- Training End --------------------------------#
