# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 23:00:20 2019

@author: hossain
"""

import pandas as pd
sp1 = 138
dp1 = 189
dp2 = 163


import csv
import datetime as ddt

#location are stored as number mapped by name 
location = [sp1,dp1,dp2]


'''
data = pd.read_csv("selected_data_weekday.csv")
data1 = data[data["dropping_point_id"]==location[1]]
data2 = data[data["dropping_point_id"]==location[2]]

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

#selected_data = pd.read_csv("TLC Trip Data/yellow_tripdata_2019-04.csv") 
selected_data = pd.read_csv("selected_data_weekday.csv") 
   
min_dur1 = 16
max_dur1 = 90
min_dur2 = 16
max_dur2 = 99
sp=0
dp=0

#for first time when file not exist
#'''
with open('selected_data_weekday-3.csv', mode='w', newline='') as file:
    #selected_data_weekday.csv file contains expected data having following field
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
    writer.writerow(['starting_point_id','dropping_point_id','pickup_datetime','dropoff_datetime','pickup_time','pickup_time_min','duration','weekday'])
#'''
                                #mode='a' for append and mode='w' for write in new file
#with open('selected_data_weekday.csv', mode='a', newline='') as file:
    #selected_data_weekday.csv file contains expected data having following field
   # writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
    #writer.writerow(['starting_point_id','dropping_point_id','pickup_time','dropoff_time','duration','pickup_time_sec','weekday'])

    it = 0
    while it<2: #for two pairs of locations
        it=it+1
        if it==1:
            sp = sp1
            dp = dp1 
            min_dur = min_dur1
            max_dur = max_dur1
        else:
            sp = sp1
            dp = dp2 
            min_dur = min_dur2
            max_dur = max_dur2
        
        data = selected_data[selected_data["starting_point_id"]==sp]
        data = data[data["dropping_point_id"]==dp]  
    
        data["pickup_time"] = pd.to_datetime(data["pickup_time"])
        data["dropoff_time"] = pd.to_datetime(data["dropoff_time"])
        
        data.to_csv("temp.csv", index=False)
    
        with open('temp.csv','rt') as f:
            data_file = csv.reader(f)
            i = 0
            for row in data_file: 
                if i==0:
                    i = i+1
                    continue
                
                a = ddt.datetime.strptime(row[2],"%Y-%m-%d %H:%M:%S") #1/5/2019 19:59
                b = ddt.datetime.strptime(row[3],"%Y-%m-%d %H:%M:%S")
                #c = a.time
                    
                diff = (b - a).total_seconds() / 60.0
                diff = int(diff)
                
                if (diff >= min_dur) and (diff <= max_dur):
                    hour = a.hour
                    minute = a.minute
                    sec = a.second
                    c = ddt.time(hour, minute,sec)
                    total_min = hour*60+minute
                    day = a.strftime("%A")
        
                    if diff < 100:
                        writer.writerow([row[0], row[1], row[2], row[3], c, total_min, diff, day])
                        
       
