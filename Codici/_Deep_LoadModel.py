import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from Classe_sismogramma_v3 import ClasseDataset

# hdf5in = '/home/silvia/Desktop/Pollino_All/Pollino_All_data_100Hz.hdf5'
# csvin = '/home/silvia/Desktop/Pollino_All/Pollino_All_metadata_100Hz.csv'
# Data_pol = ClasseDataset()
# Data_pol.leggi_custom_dataset(hdf5in, csvin)
# sample_train = len(Data_pol.sismogramma)


hdf5all = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s_Normalizzate.hdf5'
csvall = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s_Normalizzate.csv'

hdf5clean = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s_Normalizzate.hdf5'
csvclean = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s_Normalizzate.csv'

Dati_ = ClasseDataset()
Dati_.leggi_custom_dataset(hdf5clean, csvclean)

lung = len(Dati_.sismogramma[0])
semi_amp = 80
pat_tent = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/'
tentativo = ["27", "42"]

estremi_test = [43, 45, 9.5, 11.8]
xtest = []
ytest_true = []
test_indici = []
for k in range(len(Dati_.sismogramma)):
    if estremi_test[0] < Dati_.metadata['source_latitude_deg'][k] < estremi_test[1] and estremi_test[2] \
            < Dati_.metadata['source_longitude_deg'][k] < estremi_test[3]:
        xtest.append(Dati_.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp])
        test_indici.append(k)  # TODO salvo posizioni tracce di test
        if Dati_.metadata["trace_polarity"][k] == "positive":
            ytest_true.append(1)
        elif Dati_.metadata["trace_polarity"][k] == "negative":
            ytest_true.append(0)


# x_pol = np.zeros((sample_train, semi_amp * 2))
# for k in range(sample_train):
#     x_pol[k] = Dati_.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp]
# y_pol_true = np.array([Dati_.metadata["trace_polarity"][_] == "positive" for _ in range(sample_train)])
# y_pol_true = y_pol_true + 0

model_clean = keras.models.load_model(pat_tent+str(tentativo[0])+'/Tentativo_'+str(tentativo[0])+'.hdf5')
model_all = keras.models.load_model(pat_tent+str(tentativo[1])+'/Tentativo_'+str(tentativo[1])+'.hdf5')
model_clean.summary()
model_all.summary()

xtest = np.array(xtest)
print("XSHAPEEEEEEEEEEEE", xtest.shape)
y_predicted = model_clean.predict(xtest)
print(y_predicted, len(y_predicted), "\n", y_predicted, len(y_predicted))
y_predicted_ok = []
for i in y_predicted:
    y_predicted_ok.append(i[0])
y_predicted_ok = np.array(y_predicted_ok)
delta_y = np.abs(ytest_true - y_predicted_ok)

Dati_test = Dati_.seleziona_indici(test_indici)
dizio_val = {"traccia": Dati_test.metadata["trace_name"], "y_Mano": ytest_true, "y_predict": y_predicted_ok,
             "delta": delta_y}
print(len(Dati_.metadata["trace_name"]), len(ytest_true), len(y_predicted_ok), len(delta_y))

datapandas_val = pd.DataFrame.from_dict(dizio_val)

pat_confronto = '/home/silvia/Documents/GitHub/primoprogetto/Confronto_Clean_All/'
datapandas_val.to_csv(pat_confronto + "/DataClean_NetClean/DataClean_NetClean.csv", index=False)

print("###################### len delta_y, test indici ##############")
print(len(delta_y), len(test_indici))

indici_errati = []
for i in range(len(delta_y)):
    if abs(delta_y[i]) >= 0.5:
        indici_errati.append(test_indici[i])
print(len(indici_errati), "###############################")
Dati_.plotta(indici_errati, semiampiezza=120, namepng='Figure', percosro_cartellla=pat_confronto+"DataClean_NetClean")


#
# y_pol_predicted = model.predict(x_pol)
# print(y_pol_predicted, len(y_pol_predicted), "\n", y_pol_predicted, len(y_pol_predicted))
# y_pol_predicted_ok = []
# for i in y_pol_predicted:
#     y_pol_predicted_ok.append(i[0])
# y_pol_predicted_ok = np.array(y_pol_predicted_ok)
# delta_y_val = np.abs(y_pol_true - y_pol_predicted_ok)
#
# dizio_val = {"traccia_val": Dati_.metadata["trace_name"], "y_Mano_pol": y_pol_true, "y_pol_predict": y_pol_predicted_ok,
#              "delta_val": delta_y_val}
# print(len(Dati_.metadata["trace_name"]), len(y_pol_true), len(y_pol_predicted_ok), len(delta_y_val))
#
# datapandas_val = pd.DataFrame.from_dict(dizio_val)
#
# datapandas_val.to_csv(pat_tent + str(tentativo) +
#                       "/Predizioni_Pollino_tentativo_" + str(tentativo) + ".csv", index=False)
