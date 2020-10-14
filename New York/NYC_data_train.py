# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 23:21:48 2020

@author: hossain
"""
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

def split_train_test(data):
    train_vald_data = data[data.Pickup_datetime.str.contains('2019-01') | data.Pickup_datetime.str.contains('2019-03')
    | data.Pickup_datetime.str.contains('2019-04') | data.Pickup_datetime.str.contains('2019-05')]
    test_data = data[data.Pickup_datetime.str.contains('2019-06')]
    
    return train_vald_data, test_data

def create_coefficient_file(max_deg):
    #for first run
    with open('NY_yellow_taxi_model_coefficient.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)   
        i = 1
        header = ['Starting_locationID','Dropping_locationID','Weekday','Degree','Intercept']
        while i <= max_deg:
            header.append('Power_'+str(i))
            i += 1
        writer.writerow(header)
     
# -------------------- Function section end ------------------------- #

# ----------------------- Define section ------------------------- #
#without chunksize  
train_validate_data = pd.read_csv("train_validate_dataset.csv", thousands=',', dtype={"Starting_locationID":int, "Dropping_locationID":int,
    "Pickup_datetime":str, "Dropoff_datetime":str,"Pickup_time_min":int,"Distance":float,"Duration":float,"Weekday":str})#chunksize=10000)

test_data = pd.read_csv("test_dataset.csv", thousands=',', dtype={"Starting_locationID":int, "Dropping_locationID":int,
   "Pickup_datetime":str, "Dropoff_datetime":str,"Pickup_time_min":int,"Distance":float,"Duration":float,"Weekday":str})


weekdays = ['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']
color = ['b','r','g','grey','m','y','k']
#b : blue. g : green. r : red. c : cyan. m : magenta. y : yellow. k : black. w : white.

deg = 2
max_deg = 20
INFINITY = 999999
num_location = 263 #total locations

i = 0
j = 0 #pair of locations

model = LinearRegression()
# -------------------------- Define section End --------------------------------#
   
# -------------------------- Training section --------------------------------#
#file where saving coefficient for each model
#create_coefficient_file(max_deg)

i = 1
while i < num_location:
    j = 1
    while j < num_location:
        if i == j:
            j += 1
            continue     
        k = 0 
        while k < 7: # for each day
            #data loaded using chunk
            #prepare data for epecific pair of locations and weekday           
            tv_data = prepare_data(train_validate_data, i, j , weekdays[k]) 
            ts_data = prepare_data(test_data, i, j , weekdays[k]) 
            #split dataset into training (80%) and validation (20%) subsets
            train_data, validate_data = train_test_split(tv_data, test_size=0.2)
            if len(train_data)<100: # very few data available
                print(j,' apx. empty')
                k = k+1
                continue
            y_train = train_data.Duration
            X_tr = np.c_[train_data.Pickup_time_min]
            
            y_validate = validate_data.Duration
            X_v = np.c_[validate_data.Pickup_time_min]
            
            #ts_data.sort_values(by="Pickup_time_min", inplace=True)
            #x_test = pd.to_datetime(ts_data.Pickup_datetime).dt.time #taking only time
            y_test = ts_data.Duration
            X_ts = np.c_[ts_data.Pickup_time_min]
                
            #training and tuning the model; tried to find best fit
            deg = 3
            min_error = INFINITY
            expected_deg = 1
            while deg < max_deg:
                # y = a_0 + a_1*x + a_2*x^2 + . . . . . + a_deg*x^deg
                # transform indenpended variable x to x, x^2, x^3, ..... , x^deg
                X_train = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_tr) 
                model.fit(X_train, y_train) # Train the model
                #validate the model
                X_validate = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_v)
                y_pred = model.predict(X_validate)

                mserr = metrics.mean_squared_error(y_validate, y_pred)
                if mserr <= min_error:
                    min_error = mserr
                    expected_deg = deg
                deg += 1
            #training the model with expected degree 
            X_train = PolynomialFeatures(degree=expected_deg, include_bias=False).fit_transform(X_tr) 
            model.fit(X_train, y_train) # Train the model
            #Testing the model with testing dataset, show rmse
            X_test = PolynomialFeatures(degree=expected_deg, include_bias=False).fit_transform(X_ts)
            y_pred = model.predict(X_test)
            mserr = metrics.mean_squared_error(y_test, y_pred)
            #saving coefficient of the model with other parameter          
            with open('NY_yellow_taxi_model_coefficient.csv', mode='a', newline='') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                row = [i, j, weekdays[k], expected_deg, model.intercept_]
                coef = model.coef_
                t = 0
                while t < expected_deg:
                    row.append(coef[t])
                    t += 1
                writer.writerow(row)
            print(j,' Expected degree :', expected_deg,'  ','MSE :',mserr)
            k += 1
        
        j += 1
    i += 1
# -------------------------- Training End --------------------------------#