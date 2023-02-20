import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from Classe_sismogramma_v3 import ClasseDataset

hdf5ross_polarity = '/home/silvia/Desktop/SCSN(Ross)/Ross_test_polarity_Normalizzate20_data.hdf5'
csvross_polarity = '/home/silvia/Desktop/SCSN(Ross)/Ross_test_polarity_Normalizzate20_metadata.csv'
Data_Ross = ClasseDataset()
Data_Ross.leggi_custom_dataset(hdf5ross_polarity, csvross_polarity)
sample_train = len(Data_Ross.sismogramma)


# hdf5all = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s_Normalizzate.hdf5'
# csvall = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s_Normalizzate.csv'
#
# hdf5clean = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s_Normalizzate.hdf5'
# csvclean = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s_Normalizzate.csv'

# Dati_ = ClasseDataset()
# Dati_.leggi_custom_dataset(hdf5in, csvin)

lung = len(Data_Ross.sismogramma[0])
semi_amp = 80
pat_tent = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/'
tentativi = [28, 39, 40, 41, 42, 45]

# estremi_test = [43, 45, 9.5, 11.8]
# xtest = []
# ytest_true = []
# test_indici = []
# for k in range(len(Dati_.sismogramma)):
#     if estremi_test[0] < Dati_.metadata['source_latitude_deg'][k] < estremi_test[1] and estremi_test[2] \
#             < Dati_.metadata['source_longitude_deg'][k] < estremi_test[3]:
#         xtest.append(Dati_.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp])
#         test_indici.append(k)  # TODO salvo posizioni tracce di test
#         if Dati_.metadata["trace_polarity"][k] == "positive":
#             ytest_true.append(1)
#         elif Dati_.metadata["trace_polarity"][k] == "negative":
#             ytest_true.append(0)


x_pol = np.zeros((sample_train, semi_amp * 2))
for k in range(sample_train):
    x_pol[k] = Data_Ross.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp]
y_pol_true = np.array([Data_Ross.metadata["trace_polarity"][_] == "positive" for _ in range(sample_train)])
y_pol_true = y_pol_true + 0
predizioni_totali = len(y_pol_true)
dizio_pred = {"tentativi": tentativi}
list_pred_giuste = []
list_pred_sbagliate = []
for tentativo in tentativi:
    model = keras.models.load_model(pat_tent+str(tentativo)+'/Tentativo_'+str(tentativo)+'.hdf5')
    model.summary()

    xtest = np.array(x_pol)
    print("XSHAPEEEEEEEEEEEE", xtest.shape)
    y_predicted = model.predict(xtest)
    print(y_predicted, len(y_predicted), "\n", y_predicted, len(y_predicted))
    y_predicted_ok = []

    for i in y_predicted:
        y_predicted_ok.append(i[0])
    y_predicted_ok = np.array(y_predicted_ok)
    delta_y = np.abs(y_pol_true - y_predicted_ok)

    # predizioni_totali = len(y_predicted_ok)
    predizioni_giuste = 0
    predizioni_errate = 0
    for i in range(len(y_predicted_ok)):
        if abs(y_pol_true[i] - y_predicted_ok[i]) < 0.5:
            predizioni_giuste = predizioni_giuste+1
        else:
            predizioni_errate = predizioni_errate+1
    list_pred_sbagliate.append(predizioni_errate)
    list_pred_giuste.append(predizioni_giuste)
    print("totali/giuste/errate tentativo "+str(tentativo), predizioni_totali, "-", predizioni_giuste,
          "-", predizioni_errate, "-",)

    dizio_val = {"traccia": Data_Ross.metadata["trace_name"], "y_Mano": y_pol_true, "y_predict": y_predicted_ok,
                 "delta": delta_y}
    print(len(Data_Ross.metadata["trace_name"]), len(y_pol_true), len(y_predicted_ok), len(delta_y))
    datapandas_val = pd.DataFrame.from_dict(dizio_val)

    datapandas_val.to_csv(pat_tent + str(tentativo) +
                          "/Predizioni_Roos_Normalizzate20_Testset_tentativo_" + str(tentativo) + ".csv", index=False)

dizio_pred["giuste"] = list_pred_giuste
dizio_pred["sbagliate"] = list_pred_sbagliate
print(predizioni_totali, dizio_pred)
