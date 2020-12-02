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
from datetime import timedelta
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

# ---------------------- calculation duration --------------------------#
def calculate_single_dur(data, source, destination, dt):
    X_ = np.array([[0.0]])
    X_[0][0] = dt.hour*60 + dt.minute
    day = dt.strftime("%A")
    
    y_pred = 0.0
    pdata = prepare_data(data, source, destination, day)
    if pdata.empty:
        return INFINITY
    else:
        deg = pdata.Degree.values[0]
        X_val = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
        
        k = 1
        y_pred = pdata.Intercept
        while k<=deg:
            y_pred = y_pred + X_val[0][k-1]*pdata['Power_'+str(k)]
            k = k+1
        if y_pred.values[0] < 0:
            return 0
    return y_pred.values[0]
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
    
# ---------------------------- Calculation End --------------------------------#

# ---------------------------- Traffic pattern --------------------------------#
def traffic_pattern(ldata, idata, sl, dl):
    weekdays = ['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']
    colors = ['b','r','g','grey','m','y','k']
    #fig = plt.figure(figsize = (8, 5))
        
    temp = np.array([0.0])
    y_pred = np.array([0.0])
    dt = datetime(2021, 1, 2, 14, 0, 0) # saturday to friday; find lcp at 2 PM 
    # input data
    x_input = pd.to_datetime(idata.Pickup_time).dt.time #used in plot
    X_ = np.c_[idata.Pickup_time_min] 
    
    for j in range(len(x_input)-1):
        y_pred = np.concatenate([y_pred,temp])
    
    i = 0
    while i<7: # for days
        # model information of respected pair of locations and weekday
        data = prepare_data(ldata, sl, dl, weekdays[i])
        #print(data.values)
        
        if data.empty:
            parent, route, duration = lcp_DA_AL(ldata, sl, dl, num_location, dt)
            print(route)
            for j in range(len(x_input)):
                y_pred[j] = 0
                
            l = len(route)-1
            while l > 0:
                data = prepare_data(ldata, route[l], route[l-1], weekdays[i]) # get model co-efficient
                # converted x to x, x^2, x^3, . . . . . , x^deg
                deg = data.Degree.values[0]
                X_input = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
                
                j = 0
                while j<len(X_input):
                    y_pred[j] = y_pred[j] + data.Intercept
                        
                    k = 1
                    while k <= deg:
                        y_pred[j] = y_pred[j] + X_input[j][k-1]*data['Power_'+str(k)]
                        k = k+1
                    j = j+1
                l = l - 1
        else:    
            # converted x to x, x^2, x^3, . . . . . , x^deg
            deg = data.Degree.values[0]
            X_input = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
            
            j = 0
            while j<len(X_input):
                y_pred[j] = data.Intercept
                    
                k = 1
                while k <= deg:
                    y_pred[j] = y_pred[j] + X_input[j][k-1]*data['Power_'+str(k)]
                    k = k+1
                j = j+1             
        # graph ploted
        plt.plot(x_input,y_pred,colors[i],label=weekdays[i])
        plt.locator_params(nbins=6,axis='x')
        i = i+1
        dt += timedelta(days=1)
    plt.xlabel('Pickup time')
    plt.ylabel('Duration (minutes)')
    plt.legend()
    plt.show()
# -------------------------- Traffic pattern End --------------------------------#
    
# --------------------- Least congested path ----------------------- #

# -------------------- Bellman Ford Algorithm -----------------------# 
def lcp_BFA_AM(loaded_data, source, destination, num_location, dt):
    # parent[i] is parent of i
    parent = np.array([0 for x in range(num_location+1)]) 
    # duration[i] is duration from source to location i+1
    duration = np.array([INFINITY for x in range(num_location)]) 
    
    # distance for all edges
    dur_array = calculate_duration_AM(loaded_data, num_location, dt)
    print('distance calculation')
    duration[source-1] = 0
    i = 0
    while i < num_location:
        j = 0
        while j < num_location:
            k = 0
            while k < num_location:
                if j == k:
                    k = k+1
                    continue
                #if (j == source-1) & (k == destination-1):
                 #   k = k+1
                  #  continue
                if duration[k] > (duration[j]+dur_array[j][k]):
                    duration[k] = duration[j]+dur_array[j][k]
                    parent[k+1] = j+1
                k = k+1
            j = j+1
        i = i+1
    print('Duration:', duration[destination-1], 'minutes')
    route = []
    p = destination
    i = 0
    while p > 0:
        route.append(p)
        p = parent[p]
        i += 1
        if i>20:
            break
    
    print('Route:', route)
    
def lcp_BFA_AL(loaded_data, source, destination, num_location, dt):
    # parent[i] is parent of i
    parent = np.array([0 for x in range(num_location+1)]) 
    # duration[i] is duration from source to location i+1
    duration = np.array([INFINITY for x in range(num_location)]) 
    
    # distance for all edges
    dur_array = calculate_duration_AM(loaded_data, num_location, dt)
    graph = construct_graph(loaded_data, num_location, dt.strftime("%A"))
    #print(dur_array)
    print('distance calculation')
    duration[source-1] = 0
    i = 0
    while i < num_location:
        j = 0
        while j < num_location:
            k = 0
            while k < len(graph[j]):
                #if (j == source-1) & (k == destination-1):
                 #   k = k+1
                  #  continue
                if duration[graph[j][k]-1] > duration[j]+dur_array[j][graph[j][k]-1]:
                    duration[graph[j][k]-1] = duration[j]+dur_array[j][graph[j][k]-1]
                    parent[graph[j][k]] = j+1
                k = k+1
            j = j+1
        i = i+1
    
    print('Duration:', duration[destination-1], 'minutes')
    route = []
    p = destination
    i = 0
    while p > 0:
        route.append(p)
        p = parent[p]
        i += 1
        if i>20:
            break
    
    print('Route:', route)
# ------------------------------- BFA End ----------------------------------#

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
# --------------------- Least congested path end -----------------------#

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

weekdays = ['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']
color = ['b','r','g','grey','m','y','k']
#b : blue. g : green. r : red. c : cyan. m : magenta. y : yellow. k : black. w : white.

source = 1 #
destination = 4#

num_location = 18 #total locations
locationID = [x for x in range(1,num_location+1)]

ID_to_zone = locationID_name_map_all(zone)
#print(ID_to_zone)

i = 0 
j = 0 # pair of location
# -------------------------- Variable section End --------------------------------#


#-------------------------- get and show result ----------------------------#

print('Comparative travel time from ', ID_to_zone[source][1],',',ID_to_zone[source][2],
      'to', ID_to_zone[destination][1],',',ID_to_zone[destination][2])
traffic_pattern(loaded_data, input_data, source, destination)

print()
print('Least Congested Path from', ID_to_zone[source][1],',',ID_to_zone[source][2],
      'to', ID_to_zone[destination][1],',',ID_to_zone[destination][2])
cur = datetime.now() # pickup datetime
cur = datetime(2020, 12, 3, 14, 0, 0) # for specific time
start = timeit.default_timer()
lcp_BFA_AM(loaded_data, source, destination, num_location, cur)
stop = timeit.default_timer()
print('Time: ', stop - start)
# Adjacency list better 
#start = timeit.default_timer()
#lcp_BFA_AL(loaded_data, source, destination, num_location, cur)
#stop = timeit.default_timer()
#print('Time: ', stop - start)
'''
final_route = []
final_dur = []
start = timeit.default_timer()
parent, route, duration = lcp_DA_AL(loaded_data, source, destination, num_location, cur)
stop = timeit.default_timer()
print('Time: ', stop - start) 
print(route)

#adding middle location
for i in range(1):
    midp = 140
    p = midp
    final_dur.append(duration[midp-1])
    i = 0
    while p > 0:
        final_route.append(p)
        p = parent[p]
        i += 1
        if i>20:
            break
    print(final_route)
    start = timeit.default_timer()
    parent, route, duration = lcp_DA_AL(loaded_data, midp, destination, num_location, cur)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    
p = destination
final_dur.append(duration[p-1])
i = 0
while p > 0:
    final_route.append(p)
    p = parent[p]
    i += 1
    if i>20:
        break
    
rlen = len(final_route)
zone_name = ID_to_zone[final_route[rlen-1]][2]
for i in range(rlen-1):
    zone_name += ', ' + ID_to_zone[final_route[rlen-i-2]][2]
    
print(final_route)
print(final_dur)
'''
