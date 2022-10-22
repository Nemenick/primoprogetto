import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from Classe_sismogramma_v3 import ClasseDataset

#
# hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s_Normalizzate.hdf5'
# csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s_Normalizzate.csv'
#
# # hdf5in = '/home/silvia/Desktop/Pollino/Pollino_100Hz_data.hdf5'
# # csvin = '/home/silvia/Desktop/Pollino/Pollino_100Hz_metadata.csv'
#
# Dati = ClasseDataset()
# Dati.leggi_custom_dataset(hdf5in, csvin)
# print("N_sismogrammi", len(Dati.sismogramma), "N_polarità", len(Dati.metadata["trace_polarity"]))
# # Dati.elimina_tacce_indici([124709])
#
# semiampiezza = 80       # TODO
# lung = len(Dati.sismogramma[0])
# x = np.zeros((len(Dati.sismogramma), semiampiezza*2))
# y = np.array([Dati.metadata["trace_polarity"][i] == "positive" for i in range(len(Dati.sismogramma))])
# y = y + 0
# pat_tent = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/'
# tentativi = [35, 36, 37, 38]
# kern_sizs = [3, 5, 7, 9]
# time_shifts = [i-10 for i in range(21)]
# predizioni = [[[], [], []] for i in range(len(tentativi))]
# for k in range(len(tentativi)):
#     tentativo = tentativi[k]
#     kern_siz = kern_sizs[k]
#     model = keras.models.load_model(pat_tent+str(tentativo)+'/Tentativo_'+str(tentativo)+'.hdf5')  # TODO
#     model.summary()
#
#     for time_shift in time_shifts:
#         for i in range(len(Dati.sismogramma)):
#             x[i] = Dati.sismogramma[i][lung // 2 - semiampiezza + time_shift:lung // 2 + semiampiezza + time_shift]
#
#         predizione = model.evaluate(x, y, batch_size=1024)
#         predizioni[k][0].append(time_shift)
#         predizioni[k][1].append(predizione[0])
#         predizioni[k][2].append(predizione[1])
#         print("predict num", time_shift, predizione)
#
#     plt.plot(predizioni[k][0], predizioni[k][1], label="Loss_Kern_size="+str(kern_siz))
# plt.legend()
# plt.show()
# # plt.savefig(path_tentativi + "/" + str(tentativo) + "/accuracy_"+str(tentativo))
# plt.clf()
#
# for k in range(len(tentativi)):
#     tentativo = tentativi[k]
#     kern_siz = kern_sizs[k]
#     plt.plot(predizioni[k][0], predizioni[k][2], label="Accuracy_Kern_size=" + str(kern_siz))
# plt.legend()
# plt.show()
# # plt.savefig(path_tentativi + "/" + str(tentativo) + "/accuracy_"+str(tentativo))
# plt.clf()
pat_tent = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/'
model = keras.models.load_model(pat_tent + str(25) + '/Tentativo_' + str(25) + '.hdf5')
model.summary()
# yp = model.predict(x)
# print(y, len(y), "\n", yp, len(yp))
# yp_ok = []
# for i in yp:
#     yp_ok.append(i[0])
# yp_ok = np.array(yp_ok)
# delta_y = np.abs(y-yp_ok)
#
# tracce_previsione_errata = []
# tracce_previsione_incerta = []
# for i in range(len(delta_y)):
#     if delta_y[i] > 0.5:
#         tracce_previsione_errata.append(i)
#     if 0.2 < delta_y[i] < 0.5:
#         tracce_previsione_incerta.append(i)
# Dati.plotta(tracce_previsione_errata, 130, "figure_previsione_errata_80_tentativo9", "/home/silvia/Desktop/Pollino")
# Dati.plotta(tracce_previsione_incerta, 130, "figure_previsione_incerta_80_tentativo9", "/home/silvia/Desktop/Pollino")  # TODO
# dizio = {"traccia": Dati.metadata["trace_name"], "y_a_Mano": y, "y_predict": yp_ok, "delta": delta_y}
# datapandas = pd.DataFrame.from_dict(dizio)
# datapandas.to_csv('/home/silvia/Desktop/Predizioni_Pollino_semiampiezza_80_tentativo9.csv', index=False)                # TODO
#
# print(type(yp))
