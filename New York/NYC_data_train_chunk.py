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
        n += 1
        print(n)
        for i in range (0, nloc):
            dl = 1
            sloc = sl + i
            while dl<(num_location+1):
                data = tv_data[(tv_data.Starting_locationID==sloc) & (tv_data.Dropping_locationID==dl)]
                data = data[['Pickup_datetime', 'Pickup_time_min', 'Duration', 'Weekday']]
                tv_chunk[i][dl].append(data)
                dl += 1
    
    n = 0
    for ts_data in ts_d:
        n += 1
        print(n)
        for i in range (0, nloc):
            sloc = sl + i
            dl = 1
            while dl<(num_location+1):
                data = ts_data[(ts_data.Starting_locationID==sloc) & (ts_data.Dropping_locationID==dl)]
                data = data[['Pickup_datetime', 'Pickup_time_min', 'Duration', 'Weekday']]
                ts_chunk[i][dl].append(data)
                dl += 1
    
    return tv_chunk, ts_chunk

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
#train_validate_data = pd.read_csv("train_validate_dataset.csv", thousands=',', dtype={"Starting_locationID":int, "Dropping_locationID":int,
#    "Pickup_datetime":str, "Dropoff_datetime":str,"Pickup_time_min":int,"Distance":float,"Duration":float,"Weekday":str})#chunksize=10000)

#test_data = pd.read_csv("test_dataset.csv", thousands=',', dtype={"Starting_locationID":int, "Dropping_locationID":int,
#   "Pickup_datetime":str, "Dropoff_datetime":str,"Pickup_time_min":int,"Distance":float,"Duration":float,"Weekday":str})


weekdays = ['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']
color = ['b','r','g','grey','m','y','k']
#b : blue. g : green. r : red. c : cyan. m : magenta. y : yellow. k : black. w : white.

deg = 2
max_deg = 20
INFINITY = 999999
num_location = 263 #total locations
chunksize = 1000000 # chunksize

i = 0
j = 0 #pair of locations
locationID = [x for x in range(1,num_location+1)]

model = LinearRegression()
sp = 138
dp1 = 163
dp2 = 189

# -------------------------- Define section End --------------------------------#

   
# -------------------------- Training section --------------------------------#
#create_coefficient_file(max_deg)

# for 1-10     13,113
sl = 251
tv_chunk, ts_chunk = prepare_data_chunk(sl, chunksize, num_location, 12)
i = sl
while i < 263:
    print(i)
    j = 1
    while j < num_location:
        if i == j:
            j += 1
            continue 
        if len(tv_chunk[i-sl][j])<= 0: # no data available
            print(j,'Empty')
            j = j+1
            continue
        #print(i,j)
        tv_df = pd.concat(tv_chunk[i-sl][j]) #prepare_data(train_validate_data, i, j , weekdays[k]) 
        ts_df = pd.concat(ts_chunk[i-sl][j]) #prepare_data(test_data, i, j , weekdays[k])  
        
        k = 0 
        while k < 7: # for each day
            #prepare data for epecific pair of locations and weekday
            tv_data = tv_df[tv_df.Weekday == weekdays[k]]
            ts_data = ts_df[ts_df.Weekday == weekdays[k]]
            
            if len(tv_data)<70: # very few data available
                #print(i,j,' apx. empty',len(tv_data))
                k = k+1
                continue
            #for l in range(0, 5):
            #split dataset into training and validation subsets
            train_data, validate_data = train_test_split(tv_data, test_size=0.2)
            
            #x_train = train_data.Pickup_datetime #dt.date for date
            y_train = train_data.Duration
            X_tr = np.c_[train_data.Pickup_time_min]
            
            #x_validate = validate_data.Pickup_datetime #dt.date for date
            y_validate = validate_data.Duration
            X_v = np.c_[validate_data.Pickup_time_min]
            
            #ts_data.sort_values(by="Pickup_time_min", inplace=True)
            #x_test = pd.to_datetime(ts_data.Pickup_datetime).dt.time #taking only time
            y_test = ts_data.Duration
            X_ts = np.c_[ts_data.Pickup_time_min]
                
            #training and tuning the model; tried to find best fit
            deg = 3
            min_error = INFINITY
            expected_deg = deg
            while deg <= max_deg:
                # y = a_0 + a_1*x + a_2*x^2 + . . . . . + a_deg*x^deg
                # transform indenpended variable x to x, x^2, x^3, ..... , x^deg
                X_train = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_tr) 
                model.fit(X_train, y_train) # Train the model
                
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
            
            #print(j,' Expected degree :', expected_deg,'  ','MSE :',mserr)
            #print('RMSE : ',rmserr)

            k += 1
        
        j += 1
    i += 1
# -------------------------- Training End --------------------------------#

'''
with open('coefficient_weekday.csv', mode='w', newline='') as file:
    # file contains expected data having following field
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
    writer.writerow(['Starting_point_id','Dropping_point_id','Weekday','Intercept','Power_0','Power_1','Power_2','Power_3','Power_4'])


with open('coefficient_weekday.csv', mode='a', newline='') as file:
    # file contains expected data having following field
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) 
    while i<num_loc:
        sloc = location[i]
        i = i+1
        j = 0
        while j<num_loc:
            k = 0
            dloc = location[j]
            j = j+1
            while k<7:
                
                data = prepare_data(selected_data,weekdays[k],sloc,dloc)
                
                if data.empty:
                    print('empty')
                    k = k+1
                    continue
                
                data.Pickup_datetime = pd.to_datetime(data.Pickup_datetime).dt.time
                
                data = data.sort_values(by=['Pickup_time_min']) #, inplace=True
                #data.Pickup_time_min = data.Pickup_time_min + i*1440
                   
                x_train = data.Pickup_datetime #dt.date for date
                y_train = data.Duration
                X = np.c_[data.Pickup_time_min] 
                #X = mpl.dates.date2num(X)
                    
                X_train = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X) 
                        # Train the model
                model.fit(X_train, y_train)
                
                intercept = model.intercept_
                coef = model.coef_
                
                writer.writerow([sloc,dloc,weekdays[k],intercept,coef[0],coef[1],coef[2],coef[3],coef[4] ])
                
                k = k+1
    
    #flag = input('Enter letter to run or press c to cancel : ')
    #if flag == 'c':
    #break

'''

'''
tv = pd.read_csv("train_validate_dataset.csv", thousands=',', dtype={"Starting_locationID":int, "Dropping_locationID":int,
    "Pickup_datetime":str, "Dropoff_datetime":str,"Pickup_time_min":int,"Distance":float,"Duration":float,"Weekday":str})

#t = pd.read_csv("test_dataset_l.csv", thousands=',', dtype={"Starting_locationID":int, "Dropping_locationID":int,
#    "Pickup_datetime":str, "Dropoff_datetime":str,"Pickup_time_min":int,"Distance":float,"Duration":float,"Weekday":str})
#loaded_data = pd.read_csv("NY_yellow_taxi_tripdata.csv", thousands=',')
sp = 138
dp1 = 189
dp2 = 163

data1 = tv[((tv.Starting_locationID==sp) | (tv.Starting_locationID==dp1) | (tv.Starting_locationID==dp2)) & 
           ((tv.Dropping_locationID==sp) | (tv.Dropping_locationID==dp1) | (tv.Dropping_locationID==dp2))]

data1.to_csv("train_validate_dataset_l.csv", index=False)
print(tv.dtypes)

data2 = t[((t.Starting_locationID==sp) | (t.Starting_locationID==dp1) | (t.Starting_locationID==dp2)) & 
           ((t.Dropping_locationID==sp) | (t.Dropping_locationID==dp1) | (t.Dropping_locationID==dp2))]
data2.to_csv("test_dataset_l.csv", index=False)
'''

