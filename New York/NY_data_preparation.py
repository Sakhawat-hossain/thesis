
#import pandas
import pandas as pd

def split_train_test(data):
    train_vald_data = data[data.Pickup_datetime.str.contains('2019-01') | data.Pickup_datetime.str.contains('2019-02') | data.Pickup_datetime.str.contains('2019-03') | data.Pickup_datetime.str.contains('2019-04') | data.Pickup_datetime.str.contains('2019-05')]
    test_data = data[data.Pickup_datetime.str.contains('2019-06')]
    
    return train_vald_data, test_data

# loaded dataset; chunksize can be used if memory can't hold full dataset 
loaded_data = pd.read_csv("NY_yellow_taxi_tripdata.csv", thousands=',', dtype={"Starting_locationID":int, "Dropping_locationID":int,
    "Pickup_datetime":str, "Dropoff_datetime":str,"Pickup_time_min":int,"Distance":float,"Duration":float,"Weekday":str}, chunksize=1000000)
    #chunksize=1000000; 1 million rows at a time

header = True
mode = 'w'
for chunk in loaded_data:
    #splited into two datasets and saved them
    train_validate_data, test_data = split_train_test(chunk)
    train_validate_data.to_csv("train_validate_dataset.csv", index=False, header=header, mode=mode)
    test_data.to_csv("test_dataset.csv", index=False, header=header, mode=mode)
    header = False # header added only at first time
    mode = 'a' # new file created at first time, then appended
 
#without chunksize
#loaded_data = pd.read_csv("NY_yellow_taxi_tripdata.csv", thousands=',', dtype={"Starting_locationID":int, "Dropping_locationID":int,
#    "Pickup_datetime":str, "Dropoff_datetime":str,"Pickup_time_min":int,"Distance":float,"Duration":float,"Weekday":str})

#train_validate_data , test_data = split_train_test(loaded_data)
#train_validate_data.to_csv("train_validate_dataset.csv")
#test_data.to_csv("test_dataset.csv")