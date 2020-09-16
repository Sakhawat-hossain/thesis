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
location = ['Azimpur Bus Stop', 'Asad Gate', 'Gabtoli', 'Uttara Sector 1', 'Mohakhali',
            'Nikunja 1', 'Shahbagh', 'Badda']
locationID = [1, 2, 3, 4, 5, 6, 7, 8]

itr = 0 
itr_num = 1 # how many times query sent for each pair
i = 0 # 
j = 0 # i,j for each pair
loc_num = len(locationID) # number of locations
N = 30
num_q = 0
print('Total location  :  ', str(loc_num))

while itr<itr_num:
    itr = itr+1
    
    with open('selected_data_new.csv', mode='a', newline='') as file: # 'a' = for append, 'w' = for new write
        #selected_data.csv
        #selected_data_weekday.csv file contains expected data having following field
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
        #writer.writerow(['Starting_point_id','Dropping_point_id','Pickup_datetime','Distance_text',
        #                 'Distance_value','Duration_text','Duration_value','Duration_in_traffic_text',
        #                'Duration_in_traffic_value','Weekday'])
    
        i = 0
        while i<loc_num:
            j = 0
            while j<loc_num:
                if i != j:
                    cur = datetime.now()
                    cur = cur+timedelta(seconds=N)
                    
                    f = 0
                    while f<1:
                        f = f+1
                        my_dist = gmaps.distance_matrix(location[i],location[j],mode='driving',departure_time=cur)#['rows'][0]['elements'][0]
                        my_dist = my_dist['rows'][0]['elements'][0]
                        print(my_dist)
                        
                        if my_dist['status']=='OK':
                            disv = my_dist['distance']['value'] # unit meters
                            dist = my_dist['distance']['text'] # unit meters
                            
                            durv = my_dist['duration_in_traffic']['value'] # unit sec
                            durt = my_dist['duration_in_traffic']['text'] # unit sec
                            
                            durvs = my_dist['duration']['value'] # unit sec
                            durts = my_dist['duration']['text'] # unit sec
                            
                            cur2 = cur.strftime('%Y-%m-%d %H:%M:%S')
                            
                            '''
                            hour = cur.hour
                            minute = cur.minute
                            total_min = hour*60+minute
                            cur_time = cur.strftime('%H:%M:%S')
                            '''
                            day = cur.strftime("%A")
                            #print(locationID[i],' ', locationID[j],' ',cur_time)
                            writer.writerow([locationID[i], locationID[j], cur2, dist, disv, durts, durvs, durt, durv, day])
                      
                            num_q = num_q+1
                        cur = cur+timedelta(days=7,minutes=f)
                j = j+1      
            i = i+1  
    print(num_q)
    if itr == itr_num: 
        print("---------- finish ---------- ")
        break
    print(itr)
    time.sleep(750)
    
    # --------- response ------- #
'''
    {'destination_addresses': ['Gabtoli, Bangladesh'], 
     'origin_addresses':         ['Azimpur, Dhaka, Bangladesh'],
     'rows': [{'elements': [{'distance': {'text': '8.8 km', 'value': 8804},
                             'duration': {'text': '23 mins','value': 1401}, 
                             'duration_in_traffic': {'text': '23 mins', 'value': 1362},
                             'status': 'OK'}]}], 'status': 'OK'}
    
    {'distance': {'text': '8.8 km', 'value': 8804}, 'duration': 
        {'text': '23 mins', 'value': 1401}, 'duration_in_traffic': 
            {'text': '23 mins', 'value': 1362}, 'status': 'OK'}
'''

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
