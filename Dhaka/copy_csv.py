# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 23:00:20 2019

@author: hossain
"""

import pandas as pd
sp1 = [187, 7]
dp1 = [187, 7, 3]


import csv
import datetime as ddt
from datetime import datetime
from datetime import timedelta 

#location are stored as number mapped by name 
#location = [187, 111, 7, 24, 3, 134]
location = [132, 138, 163, 189]

'''
data = pd.read_csv("selected_data_weekday.csv")
data1 = data[data["dropping_point_id"]==location[1]]

print(data1.duration.max())
print(data1.duration.min())
a=int(data1.duration.mean())
print(a)
a=int(data1.duration.std())
print(a)
'''   
  
#TLC Trip Data/TLC Trip Data/yellow_tripdata_2019-02.csv

#selected_data = pd.read_csv("TLC Trip Data/yellow_tripdata_2019-05.csv") 
#selected_data = pd.read_csv("test_data_weekday.csv") 
   
min_dur1 = 16
max_dur1 = 90
min_dur = 16
max_dur = 99
sp=0
dp=0
cur = datetime.now()
cur2 = cur.strftime('%Y-%m-%d %H:%M:%S')
print(cur2)
# input data
with open('selected_data.csv', mode='r', newline='') as file:
    #selected_data_weekday.csv file contains expected data having following field
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
    
    with open('selected_data_new.csv','rt') as f:
        data_file = csv.reader(f)
        i = 0
        for row in data_file: 
            if i==0:
                i = i+1
                continue
            print(row[2])
            time = ddt.datetime.strptime(row[2],"%Y-%m-%d %H:%M:%S")
            t_min = time.hour*60 + time.minute
            #print(time, ' ', t_min)
            #writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]])
            
    print("------------ Finish ---------------")
    
#for first time when file not exist
'''
with open('selected_data_weekday.csv', mode='w', newline='') as file:
    #selected_data_weekday.csv file contains expected data having following field
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
    writer.writerow(['starting_point_id','dropping_point_id','pickup_datetime','dropoff_datetime','pickup_time','pickup_time_min','distanse','duration','weekday'])
'''
'''                                #mode='a' for append and mode='w' for write in new file
with open('selected_data_weekday.csv', mode='a', newline='') as file:
    #selected_data_weekday.csv file contains expected data having following field
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
    #writer.writerow(['starting_point_id','dropping_point_id','pickup_time','dropoff_time','duration','pickup_time_sec','weekday'])

    it1 = 0
    while it1<4: #for two pairs of locations
        #
        if it1==1:
            sp = sp1
            dp = dp1 
            min_dur = min_dur1
            max_dur = max_dur1
        else:
            sp = sp1
            dp = dp2 
            min_dur = min_dur2
            max_dur = max_dur2
        #
        sp = location[it1]
        it1 = it1+1
        it2 = 0
        while it2<4:
            dp = location[it2]
            it2 = it2+1
            
            data = selected_data[selected_data["PULocationID"]==sp]
            data = data[data["DOLocationID"]==dp]  
        
            data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"])
            data["tpep_dropoff_datetime"] = pd.to_datetime(data["tpep_dropoff_datetime"])
            
            data.to_csv("temp.csv", index=False)
    
            with open('temp.csv','rt') as f:
                data_file = csv.reader(f)
                i = 0
                for row in data_file: 
                    if i==0:
                        i = i+1
                        continue
                    
                    a = ddt.datetime.strptime(row[1],"%Y-%m-%d %H:%M:%S") #1/5/2019 19:59
                    b = ddt.datetime.strptime(row[2],"%Y-%m-%d %H:%M:%S")
                    #c = ddt.datetime.strptime(row[1],"%H:%M:%S")
                        
                    diff = (b - a).total_seconds() / 60.0
                    diff = int(diff)
                    
                    if (diff >= min_dur) and (diff <= max_dur):
                        hour = a.hour
                        minute = a.minute
                        total_min = hour*60+minute
                        sec = a.second
                        c = ddt.time(hour, minute,sec)
                        day = a.strftime("%A")
                
                        #if diff < 100:
                        writer.writerow([row[7], row[8], row[1], row[2], c, total_min, row[4], diff, day])
                      
'''       