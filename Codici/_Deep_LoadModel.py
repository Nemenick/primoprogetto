import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from Classe_sismogramma_v3 import ClasseDataset

# hdf5_predicting = '/home/silvia/Desktop/SCSN(Ross)/Ross_test_polarity_Normalizzate20_New1-1_data.hdf5'
# csv_predicting = '/home/silvia/Desktop/SCSN(Ross)/Ross_test_polarity_Normalizzate20_New1-1_metadata.csv'

# hdf5_predicting = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s_Normalizzate_New1-1.hdf5'
# csv_predicting = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s_Normalizzate_New1-1.csv'

# hdf5_predicting = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/data_U_class34.hdf5'
# csv_predicting = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/metadata_U_class34.csv'

# hdf5_predicting = '/home/silvia/Desktop/Hara/Test/Hara_test_data_Normalizzate_1-1.hdf5'
# csv_predicting = '/home/silvia/Desktop/Hara/Test/Hara_test_metadata_Normalizzate_1-1.csv'

hdf5_predicting = "/home/silvia/Desktop/Pollino_All/Pollino_All_data_100Hz_normalizzate_New1-1.hdf5"
csv_predicting = "/home/silvia/Desktop/Pollino_All/Pollino_All_metadata_100Hz_normalizzate_New1-1.csv"

Data_predicting = ClasseDataset()
Data_predicting.leggi_custom_dataset(hdf5_predicting, csv_predicting)
sample_train = len(Data_predicting.sismogramma)

lung = len(Data_predicting.sismogramma[0])
semi_amp = 75
pat_tent = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/'
tentativo = 83
salva_predizioni = False
nome_predizione = "/Predizioni_Instance_test_L_tentativo_"
# TODO predict Instance Test
"""
estremi_test = [43, 45, 9.5, 11.8]
xtest = []
y_test_true = []
test_indici = []
for k in range(len(Data_predicting.sismogramma)):
    if estremi_test[0] < Data_predicting.metadata['source_latitude_deg'][k] < estremi_test[1] and estremi_test[2] \
            < Data_predicting.metadata['source_longitude_deg'][k] < estremi_test[3]:
        xtest.append(Data_predicting.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp])
        test_indici.append(k)  # TODO salvo posizioni tracce di test
        if Data_predicting.metadata["trace_polarity"][k] == "positive":
            y_test_true.append(1)
        elif Data_predicting.metadata["trace_polarity"][k] == "negative":
            y_test_true.append(0)
Data_predicting = Data_predicting.seleziona_indici(test_indici)
"""

# TODO predict other than Instance
# """
xtest = np.zeros((sample_train, semi_amp * 2))
for k in range(sample_train):
    xtest[k] = Data_predicting.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp]
y_test_true = np.array([Data_predicting.metadata["trace_polarity"][_] == "positive" for _ in range(sample_train)])
y_test_true = y_test_true + 0
# """
model = keras.models.load_model(pat_tent+str(tentativo)+'/Tentativo_'+str(tentativo)+'.hdf5')
model.summary()

xtest = np.array(xtest)
print("XSHAPEEEEEEEEEEEE", xtest.shape)
y_predicted = model.predict(xtest, batch_size=2048)
print(y_predicted, len(y_predicted), "\n", y_predicted, len(y_predicted))
y_predicted_ok = []

for i in y_predicted:
    y_predicted_ok.append(i[0])
y_predicted_ok = np.array(y_predicted_ok)
delta_y = np.abs(y_test_true - y_predicted_ok)

predizioni_totali = len(y_predicted_ok)
predizioni_giuste = 0
predizioni_errate = 0
for i in range(len(y_predicted_ok)):
    if abs(y_test_true[i] - y_predicted_ok[i]) < 0.5:
        predizioni_giuste = predizioni_giuste+1
    else:
        predizioni_errate = predizioni_errate+1
print("totali/giuste/errate: ", predizioni_totali, "-", predizioni_giuste, "-", predizioni_errate, "-")
print(predizioni_giuste/predizioni_totali)

dizio_val = {"traccia": Data_predicting.metadata["trace_name"], "y_Mano": y_test_true, "y_predict": y_predicted_ok,
             "delta": delta_y}
print(len(Data_predicting.metadata["trace_name"]), len(y_test_true), len(y_predicted_ok), len(delta_y))
#
# dizio_val = {"traccia": Data_predicting.metadata["trace_name"], "y_Mano": y_test_true, "y_predict": y_predicted_ok,
#              "delta": delta_y}
# print(len(Data_predicting.metadata["trace_name"]), len(y_test_true), len(y_predicted_ok), len(delta_y))
datapandas_val = pd.DataFrame.from_dict(dizio_val)
# TODO cambia nome del file
if salva_predizioni:
    datapandas_val.to_csv(pat_tent + str(tentativo) +
                      nome_predizione + str(tentativo) + ".csv", index=False)
# TODO cambia se vuoi inserire la predizione nel file di metadata
# Data_predicting.metadata["Pred_tent_" + str(tentativo)] = y_predicted_ok
# Data_predicting.crea_custom_dataset(hdf5_predicting, csv_predicting)

# pat_confronto = '/home/silvia/Documents/GitHub/primoprogetto/Confronto_Clean_All/'
# datapandas_val.to_csv(pat_confronto + "/DataClean_NetClean/DataClean_NetClean.csv", index=False)


# indici_errati = []
# for i in range(len(delta_y)):
#     if abs(delta_y[i]) >= 0.5:
#         indici_errati.append(test_indici[i])
# print(len(indici_errati), "###############################")
# Dati_.plotta(indici_errati, semiampiezza=120, namepng='Figure', percosro_cartellla=pat_confronto+"DataClean_NetClean")


#
# y_pol_predicted = model.predict(x_pol)
# print(y_pol_predicted, len(y_pol_predicted), "\n", y_pol_predicted, len(y_pol_predicted))
# y_pol_predicted_ok = []
# for i in y_pol_predicted:
#     y_pol_predicted_ok.append(i[0])
# y_pol_predicted_ok = np.array(y_pol_predicted_ok)
# delta_y_val = np.abs(y_pol_true - y_pol_predicted_ok)
#
# dizio_val = {"traccia_val": Dati_.metadata["trace_name"], "y_Mano_pol": y_pol_true,
#                       "y_pol_predict": y_pol_predicted_ok, "data_val": delta_y_val}
# print(len(Dati_.metadata["trace_name"]), len(y_pol_true), len(y_pol_predicted_ok), len(delta_y_val))
#
# datapandas_val = pd.DataFrame.from_dict(dizio_val)
#
# datapandas_val.to_csv(pat_tent + str(tentativo) +
#                       "/Predizioni_Pollino_tentativo_" + str(tentativo) + ".csv", index=False)
