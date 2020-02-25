# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 23:00:20 2019

@author: hossain
"""

import pandas as pd
sp = 138
dp1 = 189
dp2 = 163


import csv
import datetime as ddt

# 4 148 123, 138 189 116
    #with open('yellow_tripdata_2019-01.csv','rt') as f:
'''
data = pd.read_csv("selected_data_1.csv", thousands=',')

#with open('selected_1.csv', mode='a', newline='') as file:
    
    #writer.writerow(['starting_point_id','dropping_point_id','pickup_time','dropoff_time','duration'])
date = "2019-01-30" #"2019-01-10"
data1 = data[data["pickup_time"].str.contains(date)]
data1.to_csv("selected_1.csv",index=False)

dd = data1.append(data1, ignore_index=True)

print(dd)
'''
'''
data = pd.read_csv("selected_data_weekday.csv")
data1 = data[data["dropping_point_id"]==dp1]
data2 = data[data["dropping_point_id"]==dp2]

print(data1.duration.max())
print(data1.duration.min())
a=int(data1.duration.mean())
print(a)
a=int(data1.duration.std())
print(a)

print(data2.duration.max())
print(data2.duration.min())
a=int(data2.duration.mean())
print(a)
a=int(data2.duration.std())
print(a)
'''   
  
#TLC Trip Data/TLC Trip Data/yellow_tripdata_2019-02.csv

selected_data = pd.read_csv("TLC Trip Data/yellow_tripdata_2019-01.csv") 
data = selected_data[selected_data["PULocationID"]==sp]
                                  #) & (selected_data.DOLocationID==dp)]
data = data[(data["DOLocationID"]==dp1) | (data["DOLocationID"]==dp2)]  
data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"])
data["tpep_dropoff_datetime"] = pd.to_datetime(data["tpep_dropoff_datetime"])

#diff = data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"]


data.to_csv("temp.csv", index=False)
   
min_dur = 16
max_dur = 90
                                #mode='a' for append
with open('test_data_weekday.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    writer.writerow(['starting_point_id','dropping_point_id','pickup_time','dropoff_time','duration','pickup_time_sec','weekday'])

    with open('temp.csv','rt') as f:
        data = csv.reader(f)
        i = 0
        for row in data: 
            if i==0:
                i = i+1
                continue
            
            a = ddt.datetime.strptime(row[1],"%Y-%m-%d %H:%M:%S") #1/5/2019 19:59
            b = ddt.datetime.strptime(row[2],"%Y-%m-%d %H:%M:%S")
                
            diff = (b - a).total_seconds() / 60.0
            diff = int(diff)
            
            if diff
            hour = a.hour
            minute = a.minute
            sec = a.second
            total_sec = hour*3600+minute*60+sec
            day = a.strftime("%A")

            if diff < 100:
                writer.writerow([row[7], row[8], row[1], row[2], diff, total_sec, day])
                    
   
