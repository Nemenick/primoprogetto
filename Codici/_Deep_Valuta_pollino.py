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
tentativo = "28"

x_pol = np.zeros((sample_train, semi_amp * 2))
for k in range(sample_train):
    x_pol[k] = Data_pol.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp]
y_pol_true = np.array([Data_pol.metadata["trace_polarity"][_] == "positive" for _ in range(sample_train)])
y_pol_true = y_pol_true + 0

model = keras.models.load_model(pat_tent+str(tentativo)+'/Tentativo_'+str(tentativo)+'.hdf5')
model.summary()


y_pol_predicted = model.predict(x_pol)
print(y_pol_predicted, len(y_pol_predicted), "\n", y_pol_predicted, len(y_pol_predicted))
y_pol_predicted_ok = []
for i in y_pol_predicted:
    y_pol_predicted_ok.append(i[0])
y_pol_predicted_ok = np.array(y_pol_predicted_ok)
delta_y_val = np.abs(y_pol_true - y_pol_predicted_ok)

dizio_val = {"traccia_val": Data_pol.metadata["trace_name"], "y_Mano_pol": y_pol_true, "y_pol_predict": y_pol_predicted_ok,
             "delta_val": delta_y_val}
print(len(Data_pol.metadata["trace_name"]),len(y_pol_true),len(y_pol_predicted_ok),len(delta_y_val))

datapandas_val = pd.DataFrame.from_dict(dizio_val)

datapandas_val.to_csv(pat_tent + str(tentativo) +
                      "/Predizioni_Pollino_tentativo_" + str(tentativo) + ".csv", index=False)
