import dask.dataframe as dd
import numpy as np
a = dd.read_csv('C:/Users/GioCar/Desktop/Simple_dataset/metadata/metadata_Instance_events_10k.csv', usecols=["source_id","station_network_code"])
print(np.array(a["source_id"]))
print(np.array(a["station_network_code"]))
# print(np.array(a["station_location_code"]))

