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

hdf5in = 'C:/Users/GioCar/Desktop/Tesi_5/data_Velocimeter_Buone_normalizzate_4s.hdf5'
csvin = 'C:/Users/GioCar/Desktop/Tesi_5/metadata_Velocimeter_Buone_normalizzate1_4s.csv'
Dati = ClasseDataset()
Dati.leggi_custom_dataset(hdf5in, csvin)
Dati.elimina_tacce_indici([124709])

semiampiezza = 130
lung = len(Dati.sismogramma[0])

x = np.zeros((len(Dati.sismogramma), semiampiezza*2))
for i in range(len(Dati.sismogramma)):
    x[i] = Dati.sismogramma[i][lung // 2 - semiampiezza:lung // 2 + semiampiezza]
y = np.array([Dati.metadata["trace_polarity"][i] == "positive" for i in range(N_test)])
y = y + 0
model = keras.models.load_model('C:/Users/GioCar/Documents/GitHub/primoprogetto/Codici/Tentativi/2/Simple_data_conv_1.0.hdf5')
model.summary()
yp = model.predict(x[0:N_test])
print(y, "\n", yp)
dizio = {"y_INGV": y, "y_predict": yp}
datapandas = pd.DataFrame.from_dict(dizio)
datapandas.to_csv('C:/Users/GioCar/Desktop/Predizioni.xls', index=False)

print(type(yp))
