# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:49:09 2019

@author: wnl
"""

# importing googlemaps module 
import googlemaps 
from datetime import datetime
from datetime import timedelta 

#import pandas as pd
import csv
import time

#open file
# Requires API key 
gmaps = googlemaps.Client(key='') 

# Requires cities name 
location = ['Azimpur', 'Gabtoli', 'Uttara Sector 1', 'Mohakhali', 'Badda']
locationID = [1, 3, 4, 5, 8]

itr = 0 
itr_num = 3 # how many times query sent for each pair
i = 0 # 
j = 0 # i,j for each pair
loc_num = 5 # number of locations
N = 30

while itr<itr_num:
    itr = itr+1
    
    with open('selected_data.csv', mode='a', newline='') as file:
        #selected_data_weekday.csv file contains expected data having following field
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
        #writer.writerow(['Starting_point_id','Dropping_point_id','Pickup_datetime','Pickup_time','Pickup_time_min','Distance','Duration','Weekday'])
        i = 0
        while i<loc_num:
            j = 0
            while j<loc_num:
                if i != j:
                    cur = datetime.now()
                    cur = cur+timedelta(seconds=N)
                    
                    my_dist = gmaps.distance_matrix(location[i],location[j],mode='driving',departure_time=cur)['rows'][0]['elements'][0]
                    print(my_dist)
                    if my_dist['status']=='OK':
                        disv = my_dist['distance']['value'] # unit meters
                        durv = my_dist['duration_in_traffic']['value'] # unit sec
                        durv = int(durv/60) # unit minute
                        cur1 = cur.strftime('%Y-%m-%d %H:%M:%S')
                        
                        hour = cur.hour
                        minute = cur.minute
                        total_min = hour*60+minute
                        cur_time = cur.strftime('%H:%M:%S')
                        day = cur.strftime("%A")
                    
                        writer.writerow([locationID[i], locationID[j], cur1, cur_time, total_min, disv, durv, day])
                    
                j = j+1      
            i = i+1  
              
    time.sleep(1800)
    
'''
i=1

while i<2:
    cur = datetime.now()
    print(cur.strftime('%Y-%m-%d %H:%M:%S'))
    cur = cur+timedelta(seconds=N)
    print(cur.strftime('%Y-%m-%d %H:%M:%S'))
    cur1 = cur.strftime('%Y-%m-%d %H:%M:%S')
                        
    hour = cur.hour
    minute = cur.minute
    total_min = hour*60+minute
    sec = cur.second
    cur_time = pd.to_datetime(cur).dt.time
    #cur_time = datetime.time(hour, minute,sec)
    day = cur.strftime("%A")
    
    
    my_dist = gmaps.distance_matrix('Azimpur','Savar',mode='driving',departure_time=cur)['rows'][0]['elements'][0]
    #bb = gmaps.distance_matrix('kkk','cc')
    # Printing the result 
    distance=my_dist['distance']['value']
    travel_time=my_dist['duration']['value']
    distancet=my_dist['distance']['text']
    travel_timet=my_dist['duration']['text']
    
    #file.write("Azimpur, Savar, %s, %d, %s, %d\r\n" %(distancet,distance,travel_timet,travel_time))

    #time.sleep(60)
    
    i=i+1
'''

#print(distance) 
#print(travel_time) 
