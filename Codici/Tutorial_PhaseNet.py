from Classe_sismogramma_v3 import ClasseDataset
import pandas as pd
import h5py
import numpy as np
# TODO reshape
percorsohdf5 = '/home/silvia/Desktop/PhaseNet_Prova/PhaseNet/test_data/data.h5'
percorsocsv = '/home/silvia/Desktop/Sample_dataset/metadata/metadata_Instance_events_10k.csv'
colonne = ['trace_name','station_code','station_channels','trace_start_time','trace_P_arrival_sample',
'trace_polarity','trace_P_uncertainty_s','source_magnitude','source_magnitude_type','source_origin_time',
'source_latitude_deg','source_longitude_deg']
filehdf5 = h5py.File(percorsohdf5, 'r')
dataset = filehdf5.get("data")
nomidata = list(dataset.keys())
print(type(nomidata), nomidata)
Data_shape = []
for i in nomidata:

    Data_shape.append(dataset.get(i))

Data_shape = np.array(Data_shape)
Data_shape[1].reshape(3,12000)
Data_reshaped = Data_shape[0].reshape(1,3,12000)
for i in range(1,len(Data_shape)):
    Data_reshaped = np.concatenate((Data_reshaped, Data_shape[i].reshape(1,3, 12000)), axis=0)

print(Data_shape.shape)
print(type(Data_shape[1]))
print(Data_reshaped.shape)
print(type(Data_reshaped[1]))