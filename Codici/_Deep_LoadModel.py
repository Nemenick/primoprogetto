import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from Classe_sismogramma_v3 import ClasseDataset
N_test = 500

# hdf5in = 'C:/Users/GioCar/Desktop/Tesi_5/data_Velocimeter_Buone_normalizzate_4s.hdf5'
# csvin = 'C:/Users/GioCar/Desktop/Tesi_5/metadata_Velocimeter_Buone_normalizzate1_4s.csv'

hdf5in = '/home/silvia/Desktop/Pollino/Pollino_100Hz_data.hdf5'
csvin = '/home/silvia/Desktop/Pollino/Pollino_100Hz_metadata.csv'

Dati = ClasseDataset()
Dati.leggi_custom_dataset(hdf5in, csvin)
print("Nsismogrammi", len(Dati.sismogramma), "Npolarità", len(Dati.metadata["trace_polarity"]))
# Dati.elimina_tacce_indici([124709])

semiampiezza = 130
lung = len(Dati.sismogramma[0])

x = np.zeros((len(Dati.sismogramma), semiampiezza*2))
for i in range(len(Dati.sismogramma)):
    x[i] = Dati.sismogramma[i][lung // 2 - semiampiezza:lung // 2 + semiampiezza]
y = np.array([Dati.metadata["trace_polarity"][i] == "positive" for i in range(len(Dati.sismogramma))])
y = y + 0
# model = keras.models.load_model('C:/Users/GioCar/Documents/GitHub/primoprogetto/Codici/Tentativi/2/Simple_data_conv_1.0.hdf5')
model = keras.models.load_model('/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/4/Tentativo_4.hdf5')
model.summary()
yp = model.predict(x)
print(y, len(y), "\n", yp, len(yp))
yp_ok = []
for i in yp:
    yp_ok.append(i[0])
dizio = {"y_a_Mano": y, "y_predict": yp_ok}
datapandas = pd.DataFrame.from_dict(dizio)
datapandas.to_csv('/home/silvia/Desktop/Predizioni_Pollino.csv', index=False)

print(type(yp))
