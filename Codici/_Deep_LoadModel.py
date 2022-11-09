import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from Classe_sismogramma_v3 import ClasseDataset

hdf5in = '/home/silvia/Desktop/Pollino_All/Pollino_All_data_100Hz.hdf5'
csvin = '/home/silvia/Desktop/Pollino_All/Pollino_All_metadata_100Hz.csv'
Data_pol = ClasseDataset()
Data_pol.leggi_custom_dataset(hdf5in, csvin)
sample_train = len(Data_pol.sismogramma)
lung = len(Data_pol.sismogramma[0])
semi_amp = 80
pat_tent = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/'
tentativo = 27

x_pol = np.zeros((sample_train * 2, semi_amp * 2))
for k in range(sample_train):
    x_pol[k] = Data_pol.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp]
    x_pol[k + sample_train] = -Data_pol.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp]
y_pol = np.array([Data_pol.metadata["trace_polarity"][_] == "positive" for _ in range(sample_train)] +
                 [1 - (Data_pol.metadata["trace_polarity"][_] == "positive") for _ in range(sample_train)])
y_pol = y_pol + 0

model = keras.models.load_model(pat_tent+str(tentativo)+'/Tentativo_'+str(tentativo)+'.hdf5')
model.summary()


a = model.evaluate(x_pol, y_pol)
print(type(a))
print(a[0], a[1])
print(a)
