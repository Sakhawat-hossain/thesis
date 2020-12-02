# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 22:30:55 2020

@author: hossain
"""
# Common imports
import numpy as np
import pandas as pd
#import csv
from datetime import datetime
import timeit

from sklearn.preprocessing import PolynomialFeatures
#from sklearn import metrics

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=15)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# -------------------- Function section ------------------------- #
INFINITY = 99999

def prepare_data(data, sl, dl, day):
    data = data[(data.Starting_locationID==sl) & (data.Dropping_locationID==dl) & (data.Weekday==day)]
    
    return data

# using Adjacency Matrix
def calculate_duration_AM(data, num_location, dt):
    X_ = np.array([[0.0]])
    dur_array = np.zeros((num_location, num_location)) #duration array
    X_[0][0] = dt.hour*60 + dt.minute
    day = dt.strftime("%A")
    
    i = 0
    while i < num_location:
        j = 0
        while j < num_location:
            if i == j:
                j = j+1
                continue
            
            y_pred = 0.0
            pdata = prepare_data(data, i+1, j+1, day)
            if pdata.empty:
                dur_array[i][j] = INFINITY
            else:
                deg = pdata.Degree.values[0]
                X_val = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
                
                k = 1
                y_pred = pdata.Intercept
                while k<=deg:
                    y_pred = y_pred + X_val[0][k-1]*pdata['Power_'+str(k)]
                    k = k+1
                dur_array[i][j] = y_pred.values[0]#.astype(int)
                if dur_array[i][j] < 0:
                    print('negative',i,j)
                    dur_array[i][j] = dur_array[i][j]*(-1)
            j = j+1
        i = i+1
    return dur_array

def construct_graph(data, num_location, day):
    graph = []
    #day = dt.strftime("%A")
    sl = 1
    while sl <= num_location:
        pdata = data[(data.Starting_locationID==sl) & (data.Weekday==day)]
        graph.append(pdata.Dropping_locationID.values)
        sl += 1
    
    return graph  

# -------------------------- Dijkstra Algorithm ----------------------------#
def lcp_DA_AL(data, source, destination, num_location, dt):
    # parent[i] is parent of i
    parent = np.array([0 for x in range(num_location+1)]) 
    # duration[i] is duration from source to location i+1
    duration = np.array([INFINITY for x in range(num_location)]) 
    # sptSet[i] track node i+1 is removed or not
    sptSet = np.array([False for x in range(num_location)]) 
    
    # distance for all edges
    dur_array = calculate_duration_AM(data, num_location, dt)
    graph = construct_graph(data, num_location, dt.strftime("%A"))
    #print(dur_array)
    #print('distance calculation')
    duration[source-1] = 0
    for i in range(num_location):
        min_dur = INFINITY
        idx = -1
        for j in range(num_location): # which node should be removed
            if (min_dur > duration[j]) and (sptSet[j] == False):
                min_dur = duration[j]
                idx = j
        #print(idx)
        if idx == -1:
            break
        sptSet[idx] = True
        for k in range(len(graph[idx])):
            if sptSet[graph[idx][k]-1] == False and duration[idx]+dur_array[idx][graph[idx][k]-1] < duration[graph[idx][k]-1]:
                duration[graph[idx][k]-1] = duration[idx]+dur_array[idx][graph[idx][k]-1]
                parent[graph[idx][k]] = idx + 1
                
    #print('Duration:', duration[destination-1], 'minutes')
    route = []
    p = destination
    i = 0
    while p > 0:
        route.append(p)
        p = parent[p]
        i += 1
        if i>20:
            break
    
    #print('Route:', route)
    return parent, route, duration
# ----------------------------- DA End ---------------------------------#

# ---------------------- Least congested time ----------------------- #
#A date is given, A range is given
#Calculate duration in every interval of 10 minutes
#should remember -> input arrival datetime, output start datetime
def least_travel_time(loaded_data, source, destination, time_range, dt):
    day = dt.strftime("%A")
    print(day)
    data = prepare_data(loaded_data, source, destination, day)
    
    start_min = time_range[0][0]*60 + time_range[0][1]
    end_min = time_range[1][0]*60 + time_range[1][1]
    time_inc = 20#10
    
    #print(end_min)
    xminute = []
    xlabel = []
    duration = []
    expt_min = 0
    expt_dur = INFINITY
    X_ = np.array([[0.0]])
    
    while start_min <= end_min:
        X_[0][0] = start_min
        if data.empty:
            print('No data availabe')
            break
        else:
            deg = data.Degree.values[0]
            X_val = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
            
            k = 1
            y_pred = data.Intercept
            while k<=deg:
                y_pred = y_pred + X_val[0][k-1]*data['Power_'+str(k)]
                k = k+1
            duration.append(y_pred.values[0])
            xminute.append(start_min)
            #hmstr = str(int(start_min/60)) + ':' + str(start_min%60)
            #xlabel.append(hmstr)
            
            if y_pred.values[0] < expt_dur:
                expt_min = start_min
                expt_dur = y_pred.values[0]
        start_min += time_inc
    #durationArr.append([start_min,y_pred.values[0]])

    print('Least Congested Time (',time_range[0][0],':',time_range[0][1],'-',time_range[1][0],':',time_range[1][1],'):')
    print('At ', int(expt_min/60), ':', str(expt_min%60), ', Duration: ', int(round(expt_dur)), ' minutes')

    #fig = plt.figure(figsize = (8, 5))
    plt.bar(xminute, duration, width=12)
    plt.xlabel('Pickup time')
    plt.ylabel('Duration (minutes)')
    arr = np.arange(xminute[0], xminute[len(xminute)-1]+1, 20)
    for i in arr:
        hmstr = str(int(i/60)) + ':' + str(i%60)
        xlabel.append(hmstr)
    plt.xticks(arr, xlabel, rotation='vertical')
    #plt.legend()
    plt.show()
# ---------------------- When range default ---------------------------#
#A date is given
#A range is default from 6 to 23.59
#Morning 6 - 10, Noon 10 - 14, Afternoon 14 - 18, Evening 18 - 22, night - rest
#Calculate duration in every interval of 30 minutes
  
def ltt_default(loaded_data, source, destination, num_location, dt):
    day = dt.strftime("%A")
    route = []
    time_range = [7*60, 10*60, 14*60, 18*60, 20*60]
    time_inc = 20
    
    #print(end_min)
    duration = []
    xlabel = []
    expt_val = []
    X_ = np.array([[0.0]])
    
    data = prepare_data(loaded_data, source, destination, day)
    if data.empty:
        parent, route, durtn = lcp_DA_AL(loaded_data, source, destination, num_location, dt)
        data = []
        for i in range(len(route)-1):
            data1 = prepare_data(loaded_data, route[len(route)-1-i], route[len(route)-2-i], day) # get model co-efficient
            data.append(data1)
        for i in range(4):
            start_min = time_range[i] + 1
            end_min = time_range[i+1]
            expt_min = 0
            expt_dur = INFINITY
            xminute = []
            yvalue = []
            j = 0
            while start_min <= end_min:  
                xminute.append(start_min)
                yvalue.append(0)
                for l in range(len(route)-1):
                    # converted x to x, x^2, x^3, . . . . . , x^deg
                    deg = data[l].Degree.values[0]
                    X_[0][0] = start_min
                    X_val = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
                
                    k = 1
                    y_pred = data[l].Intercept
                    while k<=deg:
                        y_pred = y_pred + X_val[0][k-1]*data[l]['Power_'+str(k)]
                        k = k+1
                    yvalue[j] += y_pred.values[0]
                duration.append([start_min,y_pred.values[0]])
                if yvalue[j] < expt_dur:
                    expt_dur = yvalue[j]
                    expt_min = start_min
                j += 1                
                start_min += time_inc
            expt_dur = int(round(expt_dur, 0))
            expt_val.append([expt_min, expt_dur])
            
            plt.bar(xminute, yvalue, width=14)
    else:    
        deg = data.Degree.values[0]
        for i in range(4):
            start_min = time_range[i] + 1
            end_min = time_range[i+1]
            expt_min = 0
            expt_dur = INFINITY
            xminute = []
            yvalue = []
            while start_min <= end_min:
                X_[0][0] = start_min
                X_val = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
                
                k = 1
                y_pred = data.Intercept
                while k<=deg:
                    y_pred = y_pred + X_val[0][k-1]*data['Power_'+str(k)]
                    k = k+1
                duration.append([start_min,y_pred.values[0]])
                xminute.append(start_min)
                yvalue.append(y_pred.values[0])
                
                if y_pred.values[0] < expt_dur:
                    expt_dur = y_pred.values[0]
                    expt_min = start_min
                
                start_min += time_inc
            expt_dur = int(round(expt_dur, 0))
            expt_val.append([expt_min, expt_dur])
            
            plt.bar(xminute, yvalue, width=14)
        #print(routes)
    print('')
    print('Least Congested Time: on', day)
    #fig = plt.figure(figsize = (8, 5))
    plt.xlabel('Pickup time')
    plt.ylabel('Duration (minutes)')
    plt.title(day)
    arr = np.arange(time_range[0], time_range[len(time_range)-1]+1, time_inc*3)
    for i in arr:
        hmstr = str(int(i/60)) + ':' + str(i%60)
        xlabel.append(hmstr)
    plt.xticks(arr, xlabel, rotation='vertical')
    #plt.legend()
    plt.show()
        
    i = 0
    while i < len(expt_val):
        if i == 0:
            print('In morning:')
        elif i == 1:
            print('In midday:')
        elif i == 2:
            print('In afternoon:')
        elif i == 3:
            print('In evening:')
        else:
            print('In night:')
        print('At ', int(expt_val[i][0]/60), ':', str(expt_val[i][0]%60), ', Duration: ',
              str(expt_val[i][1]), ' minutes')
        i = i+1    
        
def ltt_seven_days_default(loaded_data, source, destination, num_location, days):
    #time_range = [6*60, 10*60, 14*60, 18*60, 22*60]
    time_range = [8*60, 10*60, 14*60, 17*60, 20*60] # 7AM to 8PM
    time_inc = 20
    dt = datetime(2021, 1, 2, 14, 0, 0) # saturday to friday; find lcp at 2 PM 
    
    for ii in range(7):
        #print(end_min)
        expt_min = 0
        expt_dur = INFINITY
        X_ = np.array([[0.0]])
        
        data = prepare_data(loaded_data, source, destination, days[ii])
        if data.empty:
            parent, route, durtn = lcp_DA_AL(loaded_data, source, destination, num_location, dt)
            data = []
            for i in range(len(route)-1):
                data1 = prepare_data(loaded_data, route[len(route)-1-i], route[len(route)-2-i], days[ii]) # get model co-efficient
                data.append(data1)
            for i in range(4):
                start_min = time_range[i] + 1
                end_min = time_range[i+1]
                xminute = []
                duration = []
                j = 0
                while start_min <= end_min:  
                    xminute.append(start_min)
                    duration.append(0)
                    for l in range(len(route)-1):
                        # converted x to x, x^2, x^3, . . . . . , x^deg
                        deg = data[l].Degree.values[0]
                        X_[0][0] = start_min
                        X_val = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
                    
                        k = 1
                        y_pred = data[l].Intercept
                        while k<=deg:
                            y_pred = y_pred + X_val[0][k-1]*data[l]['Power_'+str(k)]
                            k = k+1
                        duration[j] += y_pred.values[0]
                    if duration[j] < expt_dur:
                        expt_dur = duration[j]
                        expt_min = start_min
                    j += 1                
                    start_min += time_inc
                #plt.bar(xminute, duration, width=14) 
            expt_dur = int(round(expt_dur, 0))       
        else:
            deg = data.Degree.values[0]    
            for i in range(4):
                start_min = time_range[i] + 1
                end_min = time_range[i+1]
                xminute = [] 
                duration = []
                while start_min <= end_min:
                    X_[0][0] = start_min
                    X_val = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
                    
                    k = 1
                    y_pred = data.Intercept
                    while k<=deg:
                        y_pred = y_pred + X_val[0][k-1]*data['Power_'+str(k)]
                        k = k+1                
                    if y_pred.values[0] < expt_dur:
                        expt_dur = y_pred.values[0]
                        expt_min = start_min
                    
                    xminute.append(start_min)
                    duration.append(y_pred.values[0])
                    start_min += time_inc
                #plt.bar(xminute, duration, width=12)
            expt_dur = int(round(expt_dur, 0))
            #duration.append([day, expt_min, expt_dur])
        
        print(days[ii], int(expt_min/60), ':', str(expt_min%60), expt_dur)               
        #fig = plt.figure(figsize = (8, 5))
        #plt.bar(xminute, duration, width=12)
        '''
        plt.xlabel('Pickup time')
        plt.ylabel('Duration (minutes)')
        plt.title(days[ii])
        arr = np.arange(time_range[0], time_range[len(time_range)-1]+1, 3*time_inc)
        xlabel = []
        for i in arr:
            hmstr = str(int(i/60)) + ':' + str(i%60)
            xlabel.append(hmstr)
        plt.xticks(arr, xlabel, rotation='vertical')
        #plt.legend()
        plt.show()
        '''
        
def ltt_seven_days(loaded_data, source, destination, days, time_range):
    #time_range = [6*60, 10*60, 14*60, 18*60, 22*60]
    time_inc = 20
    X_ = np.array([[0.0]])
    
    for i in range(7):
        xminute = []
        xlabel = []
        start_min = time_range[0][0]*60 + time_range[0][1]
        end_min = time_range[1][0]*60 + time_range[1][1]
        expt_min = 0
        expt_dur = INFINITY
        duration = []
        data = prepare_data(loaded_data, source, destination, days[i])
        if data.empty:
            print('No data availabe')
            continue
        
        while start_min <= end_min:
            X_[0][0] = start_min
            deg = data.Degree.values[0]
            X_val = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
            
            k = 1
            y_pred = data.Intercept
            while k<=deg:
                y_pred = y_pred + X_val[0][k-1]*data['Power_'+str(k)]
                k = k+1
            duration.append(y_pred.values[0])
            xminute.append(start_min)
            if y_pred.values[0] < expt_dur:
                expt_min = start_min
                expt_dur = y_pred.values[0]
            start_min += time_inc
    
        #print('Least Congested Time (',time_range[0][0],':',time_range[0][1],'-',time_range[1][0],':',time_range[1][1],'):')
        print('On',days[i], 'At ', int(expt_min/60), ':', str(expt_min%60), ', Duration: ', int(round(expt_dur)), ' minutes')
        '''
        #fig = plt.figure(figsize = (8, 5))
        plt.bar(xminute, duration, width=12)
        plt.xlabel('Pickup time')
        plt.ylabel('Duration (minutes)')
        arr = np.arange(xminute[0], xminute[len(xminute)-1]+1, time_inc)
        for i in arr:
            hmstr = str(int(i/60)) + ':' + str(i%60)
            xlabel.append(hmstr)
        plt.xticks(arr, xlabel, rotation='vertical')
        #plt.legend()
        plt.show()
        '''
# --------------------------- Default range End --------------------------------#

# ---------------------------- Traffic pattern --------------------------------#
def traffic_pattern_all(ldata, idata, sl, dl):
    weekdays = ['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']
    colors = ['b','r','g','grey','m','y','k']
    #fig = plt.figure(figsize = (7, 5))
    
    max_min = 0
    
    i = 0
    while i<7: # for days
        # model information of respected pair of locations and weekday
        data = prepare_data(ldata, sl, dl, weekdays[i])
        #print(data.values)
        if data.empty:
            i += 1
            continue
        
        # input data
        x_input = []#pd.to_datetime(idata.Pickup_time).dt.time #used in plot
        xmin = 0
        X_ = np.c_[idata.Pickup_time_min] 
        # converted x to x, x^2, x^3, . . . . . , x^deg
        deg = data.Degree.values[0]
        X_input = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
        
        y_pred = np.array([0.0])
        temp = np.array([0.0])
        j = 0
        while j<len(X_input):
            max_min = xmin+i*1450
            x_input.append(xmin+i*1450)
            xmin += 15
            if j==0:
                y_pred[0] = data.Intercept
            else:
                temp[0] = data.Intercept
                y_pred = np.concatenate([y_pred,temp])
                
            k = 1
            while k <= deg:
                y_pred[j] = y_pred[j] + X_input[j][k-1]*data['Power_'+str(k)]
                k = k+1
            j = j+1             
        # graph ploted
        plt.plot(x_input,y_pred,colors[i],label=weekdays[i])
        plt.locator_params(nbins=6,axis='x')
        i = i+1
    #
    arr = np.arange(0, max_min+1460, 1450)
    xlabel = ['00:00']
    temp = 23
    for i in range(7):
        strr = str(temp) + ':' + '59'
        xlabel.append(strr)
        temp += 24
        
    plt.xticks(arr, xlabel)
    plt.xlabel('Pickup time')
    plt.ylabel('Duration (minutes)')
    plt.legend()
    plt.show()
# -------------------------- Traffic pattern End --------------------------------#
    
# ------------------------- Least congested time end -------------------------- #   
def locationID_name_map_all(zone):
    mapzone = [[]]
    for i in range(len(zone)):
        mr = zone[zone.LocationID == (i+1)]
        mr = mr.values
        #print(mr)
        temp = [mr[0][0], mr[0][1], mr[0][2]]
        mapzone.append(temp)
        
    return mapzone
    
def locationID_name_map(zone, locationID):
    maprow = zone[zone.LocationID == locationID]
    
    return maprow
    
# -------------------------- Function section end ---------------------------- #


# ----------------------- Variable section ------------------------- #
#loaded required file
loaded_data = pd.read_csv("Dhaka_model_coefficient.csv", thousands=',')
input_data = pd.read_csv("input_data.csv", thousands=',') #used in plot
input_data = input_data[(input_data.Pickup_time_min > 60*7) & (input_data.Pickup_time_min < 60*22)] # 7 AM to 8PM
zone = pd.read_csv("zone_lookup.csv", thousands=',') #LocationID to Location name mapping

weekdays = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday', 'Saturday']
color = ['b','r','g','grey','m','y','k']
#b : blue. g : green. r : red. c : cyan. m : magenta. y : yellow. k : black. w : white.

source = 1 #
destination = 4#

num_location = 18 #total locations
locationID = [x for x in range(1,num_location+1)]

ID_to_zone = locationID_name_map_all(zone)
#print(ID_to_zone)

# -------------------------- Variable section End --------------------------------#


#-------------------------- get and show result ----------------------------#

print('Comparative travel time from ', ID_to_zone[source][1],',',ID_to_zone[source][2],
      'to', ID_to_zone[destination][1],',',ID_to_zone[destination][2])
#traffic_pattern(loaded_data, input_data, source, destination)

print()
#A date is given
#A range is default from 6 to 23.59
#Morning 6 - 10, Noon 10 - 14, Afternoon 14 - 18, Evening 18 - 22, night - rest
#Calculate duration in every interval of 30 minutes

print('Least Congested Time for', ID_to_zone[source][1],',',ID_to_zone[source][2],
      'to', ID_to_zone[destination][1],',',ID_to_zone[destination][2])
time_range = [[8, 00], [14, 0]] #[hour, minute]
cur = datetime(2020, 11, 22, 10, 0, 0) #datetime.now() # pickup datetime
#least_travel_time(loaded_data, source, destination, time_range, cur)
# default
print('')
cur = datetime(2020, 12, 22, 10, 0, 0)
#ltt_default(loaded_data, source, destination, num_location, cur)

ltt_seven_days_default(loaded_data, source, destination, num_location, weekdays)

#ltt_seven_days(loaded_data, source, destination, weekdays, time_range)

#traffic_pattern_all(loaded_data, input_data, source, destination)
