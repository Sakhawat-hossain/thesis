# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 19:04:13 2020

@author: hossain
"""

# Common imports
import heapq
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

import itertools

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=18)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=10)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# ----------------------- Define section ------------------------- #

selected_data = pd.read_csv("coefficient_weekday.csv", thousands=',')
#input_data = pd.read_csv("input_data.csv", thousands=',')

weekdays = ['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']
color = ['b','r','g','grey','m','y','k']
#b : blue. g : green. r : red. c : cyan. m : magenta. y : yellow. k : black. w : white.

deg = 8
INFINITY = 999999

source = 1
destination = 4

num_location = 8 #total locations
locationID = [x for x in range(1,num_location+1)]

# -------------------------- End --------------------------------#

def prepare_data(send_data,day,sp,dp):
    send_data = send_data[(send_data.Starting_point_id==sp) & (send_data.Dropping_point_id==dp) & (send_data.Weekday==day)]
   
    return send_data

# ---------------------- calculate duration --------------------------#
def calculate_duration(cur):
    i = 0
    X_ = np.array([[0.0]])
    dur_array = np.zeros((num_location, num_location)) #duration array
    
    while i < num_location:
        j = 0
        while j < num_location:
            if i == j:
                j = j+1
                continue
            
            X_[0][0] = cur.hour*60 + cur.minute
            y_pred = 0.0
            day = cur.strftime("%A")
            data = prepare_data(selected_data, day, locationID[i], locationID[j])
            if data.empty:
                #print('empty')
                dur_array[i][j] = INFINITY
            else:
                X_val = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X_)
                
                k = 0
                y_pred = data.Intercept
                while k<deg:
                    y_pred = y_pred + X_val[0][k]*data['Power_'+str(k)]
                    k = k+1
                dur_array[i][j] = y_pred.values[0]
            #dur_array[i][j] = y_pred.values[0]
            j = j+1
        i = i+1
    #print(dur_array)
    return dur_array
# ---------------------------- End --------------------------------#

# ---------------------- When range given ---------------------------#
'''
A date is given
A range is given
Calculate duration in every interval of 30 minutes

'''   
# -------------------- Bellman Ford Algorithm ---------------------# 
start_r = 8
end_r = 12
routes = []
cur = datetime(2020, 9, 13, start_r, 0, 0) # for specific time

end_min = end_r*60
itr = 0
while itr<30:
    itr = itr+1
    parent = np.array([0 for x in range(num_location+1)]) # parent[i] is parent of i
    duration = np.array([INFINITY for x in range(num_location)]) # duration[i] is dur of loc i+1

    dur_array = calculate_duration(cur)
    
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
                if (j == source-1) & (k == destination-1):
                    k = k+1
                    continue
                if duration[k] > duration[j]+dur_array[j][k]:
                    duration[k] = duration[j]+dur_array[j][k]
                    parent[k+1] = j+1
                k = k+1
            j = j+1
        i = i+1    
    #print(duration)
    #print(parent)

    route = []
    i = 0
    p = destination
    while i<num_location:
        route.append(p)
        p = parent[p]
        if p==source:
            route.append(p)
            break
        i = i+1
    route.append(duration[destination-1])
    routes.append(route)
    cur = cur + timedelta(minutes=30)
    
    if (cur.hour*60+cur.minute) > end_min:
        break

print(routes)
i = 0
temp = INFINITY
idx = -1
while i < len(routes):
    j = len(routes[i])-1
    if temp > routes[i][j]:
        temp = routes[i][j]
        idx = i
    i = i+1
print('Least Congested Time (', start_r, '-', end_r,'):')
t = start_r*60 + 30*idx
print('At ', int(t/60), ':', str(t%60), ', Duration: ', str(routes[idx][len(routes[idx])-1]), ' minutes')

# --------------------------- End --------------------------------#


# ---------------------- When range default ---------------------------#
'''
A date is given
A range is default from 6 to 23.59
Morning 6 - 10, Noon 10 - 14, Afternoon 14 - 18, Evening 18 - 22, night - rest
Calculate duration in every interval of 30 minutes

'''  
# -------------------- Bellman Ford Algorithm ---------------------# 
start_r = 6
end_r = 23
routes = []
cur = datetime(2020, 9, 15, start_r, 30, 0) # for specific time

end_min = end_r*60
itr = 0
while itr<40:
    itr = itr+1
    parent = np.array([0 for x in range(num_location+1)]) # parent[i] is parent of i
    duration = np.array([INFINITY for x in range(num_location)]) # duration[i] is dur of loc i+1

    dur_array = calculate_duration(cur)
    
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
                if (j == source-1) & (k == destination-1):
                    k = k+1
                    continue
                if duration[k] > duration[j]+dur_array[j][k]:
                    duration[k] = duration[j]+dur_array[j][k]
                    parent[k+1] = j+1
                k = k+1
            j = j+1
        i = i+1    

    route = []
    i = 0
    p = destination
    while i<num_location:
        route.append(p)
        p = parent[p]
        if p==source:
            route.append(p)
            break
        i = i+1
    route.append(duration[destination-1])
    routes.append(route)
    cur = cur + timedelta(minutes=30)
    
    if (cur.hour*60 + cur.minute) > end_min:
        break
#print(routes)
print('')
print('Least Congested Time:')
i = 0
temp = INFINITY
idx = -1
while i < len(routes):
    j = len(routes[i])-1
    if temp > routes[i][j]:
        temp = routes[i][j]
        idx = i
    if i%8 == 0:
        t = start_r*60 + 30*idx + 30
        if t <= 60*10:
            print('In morning:')
        elif t <= 60*14:
            print('In midday:')
        elif t <= 60*18:
            print('In afternoon:')
        elif t <= 60*22:
            print('In evening:')
        else:
            print('In night:')
        print('At ', int(t/60), ':', str(t%60), ', Duration: ', str(routes[idx][len(routes[idx])-1]), ' minutes')
        temp = INFINITY
    i = i+1
# --------------------------- End --------------------------------#

'''
# ---------------------- Dijkstra Algorithm ----------------------# 
        # --------------------- Heap --------------------------#
route = []
parent = np.zeros(num_location+1) # parent[i] is parent of i
min_heap = []
#heapq.heapify(min_heap)

#pq = []                         # list of entries arranged in a heap
entry_finder = {}               # mapping of tasks to entries
REMOVED = '<removed-task>'      # placeholder for a removed task
counter = itertools.count()     # unique sequence count

def add_task(task, priority=0):
    #'Add a new task or update the priority of an existing task'
    if task in entry_finder:
        remove_task(task)
    count = next(counter)
    entry = [priority, count, task]
    entry_finder[task] = entry
    heapq.heappush(min_heap, entry)

def remove_task(task):
    #'Mark an existing task as REMOVED.  Raise KeyError if not found.'
    entry = entry_finder.pop(task)
    entry[-1] = REMOVED

def pop_task():
    #'Remove and return the lowest priority task. Raise KeyError if empty.'
    while min_heap:
        priority, count, task = heapq.heappop(min_heap)
        if task is not REMOVED:
            del entry_finder[task]
        return task, priority
    #raise KeyError('pop from an empty priority queue')

i = 1
while i <= num_location:
    if i == source:
        #heapq.heappush(min_heap, (0, i))
        add_task(source, 0)
    else:
        #heapq.heappush(min_heap, (INFINITY, i))
        add_task(i, INFINITY)
    i = i+1
print(min_heap)
#idx, dur = pop_task()

# ------------------------- End -----------------------------#

while len(min_heap) > 0:

    idx, dur = pop_task()
    #print(min_heap)
    if idx is REMOVED:
        break
    route.append(idx)
    
    i = 0
    while i < num_location:
        if locationID[i] == idx: # already removed
            i = i+1
            continue
        if i+1 in entry_finder: # task => i+1
            val = dur + dur_array[idx-1][i]
            event = entry_finder[i+1]
            #print(event[0])
            if val < event[0]:
                add_task(i+1, val)
                parent[i] = idx
        i = i+1
    print(parent)
    #print(min_heap)
print(route)
# ------------------------ End --------------------------#
'''
