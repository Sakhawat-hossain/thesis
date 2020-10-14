# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 23:00:20 2019

@author: hossain
"""

import pandas as pd
import csv
import datetime as ddt
from datetime import timedelta 
import seaborn as sbn
sbn.set()

import matplotlib as mpl
import matplotlib.pyplot as plt   

with open('NY_yellow_taxi_tripdata.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
    writer.writerow(['Starting_locationID','Dropping_locationID','Pickup_datetime','Dropoff_datetime','Pickup_time_min','Distance','Duration','Weekday'])
    
    with open('yellow_tripdata_2019-01.csv','rt') as f:
        data_file = csv.reader(f)
        i = 0
        for row in data_file: 
            if i==0: # escape first row
                i = i+1
                continue 
            if float(row[4])<=0: # when distance <= 0
                continue
            # convert string to datetime formate
            a = ddt.datetime.strptime(row[1],"%Y-%m-%d %H:%M:%S")
            b = ddt.datetime.strptime(row[2],"%Y-%m-%d %H:%M:%S")
            day = a.strftime("%A")
            # mapping time to integer
            total = a.hour*60 + a.minute
                
            diff = (int) (b - a).total_seconds() / 60.0
            # duration negative or zero; or when duration > 300 minutes
            if (diff <= 0) | (diff > 300): 
                continue
            writer.writerow([row[7], row[8], row[1], row[2], total, row[4], diff, day])     
            