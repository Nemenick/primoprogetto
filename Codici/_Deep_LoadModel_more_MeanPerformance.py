import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from Classe_sismogramma_v3 import ClasseDataset
# houtu = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/Dati_sotto_1_perc/data_up_sotto_1_perc.hdf5'
# coutu = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/Dati_sotto_1_perc/metadata_up_sotto_1_perc.csv'
# # houtd = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/Dati_sotto_1_perc/data_down_sotto_1_perc.hdf5'
# # coutd = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/Dati_sotto_1_perc/metadata_down_sotto_1_perc.csv'
# houtd = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/Dati_sotto_1_perc/data_down_sotto_1_perc.hdf5'
# coutd = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/Dati_sotto_1_perc/metadata_down_sotto_1_perc.csv'

# hdf5_predicting = '/home/silvia/Desktop/SCSN(Ross)/Ross_test_polarity_Normalizzate20_New1-1_data.hdf5'
# csv_predicting = '/home/silvia/Desktop/SCSN(Ross)/Ross_test_polarity_Normalizzate20_New1-1_metadata.csv'

# hdf5_predicting = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/data_U_class34.hdf5'
# csv_predicting = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/metadata_U_class34.csv'

# hdf5_predicting = '/home/silvia/Desktop/Hara/Test/Hara_test_data_Normalizzate_1-1.hdf5'
# csv_predicting = '/home/silvia/Desktop/Hara/Test/Hara_test_metadata_Normalizzate_1-1.csv'

# hdf5_predicting = "/home/silvia/Desktop/Pollino_All/Pollino_All_data_100Hz_normalizzate_New1-1.hdf5"
# csv_predicting = "/home/silvia/Desktop/Pollino_All/Pollino_All_metadata_100Hz_normalizzate_New1-1.csv"

# hdf5_predicting = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/data_D_class47_54.hdf5'
# csv_predicting = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/metadata_D_class47_54.csv'
# hdf5_predicting = houtu
# csv_predicting = coutu

# hdf5_predicting = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s_Normalizzate_New1-1.hdf5'
# csv_predicting = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s_Normalizzate_New1-1.csv'

# hdf5_predicting ='/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s_Normalizzate_New1-1.hdf5'
# csv_predicting = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s_Normalizzate_New1-1.csv'

# hdf5_predicting = f'/home/silvia/Desktop/SCSN(Ross)/Normaliz/Ross_test_polarity_data_Normalizzate{str(asoglia)}_New1-1.hdf5'
# csv_predicting = f'/home/silvia/Desktop/SCSN(Ross)/Normaliz/Ross_test_polarity_metadata_Normalizzate{str(asoglia)}_New1-1.csv'

hdf5_predicting = "/home/silvia/Desktop/Instance_Data/Undecidable/Instance_undecidable_data_tot_no0.hdf5"
csv_predicting = "/home/silvia/Desktop/Instance_Data/Undecidable/Instance_undecidable_metadata_tot_no0.csv"

Data_predicting = ClasseDataset()
Data_predicting.leggi_custom_dataset(hdf5_predicting, csv_predicting)
sample_train = len(Data_predicting.sismogramma)

lung = len(Data_predicting.sismogramma[0])
semi_amp = 80
tentativo = "8"                # More_10
path_tentativi = f'/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/More_{tentativo}'
salva_predizioni = False
time_shift = 0
nome_predizione = "/Predizioni_Instance_Pol_Test"

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
    xtest[k] = Data_predicting.sismogramma[k][lung // 2 - semi_amp + time_shift:lung // 2 + semi_amp + time_shift]
y_test_true = np.array([Data_predicting.metadata["trace_polarity"][_] == "positive" for _ in range(sample_train)])
y_test_true = y_test_true + 0
# """
xtest = np.array(xtest)

list_acc_test = []
for it in os.scandir(path_tentativi):
    if it.is_dir():
        print(it.path)          # it.path == /home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/More_1/5
        tent = it.path[-1]
        model = keras.models.load_model(it.path + f"/Tentativo_More_{tentativo}_" + str(tent)+".hdf5")
        model.summary()

        
        yp_test = model.predict(xtest, batch_size=4096)
        
        # print(y_test_true, len(y_test_true), "\n", yp_test, len(yp_test))
        yp_ok_test = []
        for i in yp_test:
            yp_ok_test.append(i[0])
        yp_ok_test = np.array(yp_ok_test)
        delta_y_test = np.abs(y_test_true - yp_ok_test)


        acc_fin_test = 0
        for delta in delta_y_test:
            if abs(delta) < 0.5:
                acc_fin_test += 1
        acc_fin_test = acc_fin_test / len(delta_y_test)


        list_acc_test.append(acc_fin_test)

        # tracce_previsione_errata = []
        # tracce_previsione_incerta = []
        # for i in range(len(delta_y)):
        #     if delta_y[i] > 0.5:
        #         tracce_previsione_errata.append(i)
        #     if 0.2 < delta_y[i] < 0.5:
        #         tracce_previsione_incerta.append(i)
        # Dati_validation.plotta(tracce_previsione_errata, 130, "figure_previsione_errata_tentativo_"+str(tentativo),
        # "/home/silvia/Desktop/Pollino")
        # Dati_validation.plotta(tracce_previsione_incerta, 130, "figure_previsione_incerta_tentativo_"+str(tentativo),
        # "/home/silvia/Desktop/Pollino")  # TODO

        dizio_test = {"traccia": Data_predicting.metadata["trace_name"], "y_Mano": y_test_true,
                    "y_predict": yp_ok_test, "delta": delta_y_test}
        datapandas_test = pd.DataFrame.from_dict(dizio_test)
        if salva_predizioni:
            datapandas_test.to_csv(f"{it.path}/{nome_predizione}_More_{tentativo}_{tent}.csv", index=False)

if salva_predizioni:
    lista_media_test, lista_std_test = ["" for i in range(len(list_acc_test))], ["" for i in range(len(list_acc_test))]
    lista_media_test[0] = np.mean(list_acc_test)
    lista_std_test[0] = np.std(list_acc_test)
    dizio_acc = {"Acc_test": list_acc_test, "Mean_test": lista_media_test, "Std_test": lista_std_test}

    datapandas_acc = pd.DataFrame.from_dict(dizio_acc)
    datapandas_acc.to_csv(f"{path_tentativi}/{nome_predizione}_More_{tentativo}.csv", index=False)







# model = keras.models.load_model(pat_tent+str(tentativo)+'/Tentativo_'+str(tentativo)+'.hdf5')
# model.summary()

# xtest = np.array(xtest)
# print("XSHAPEEEEEEEEEEEE", xtest.shape)
# y_predicted = model.predict(xtest, batch_size=2048)
# print(y_predicted, len(y_predicted), "\n", y_predicted, len(y_predicted))
# y_predicted_ok = []

# for i in y_predicted:
#     y_predicted_ok.append(i[0])
# y_predicted_ok = np.array(y_predicted_ok)
# delta_y = np.abs(y_test_true - y_predicted_ok)

# predizioni_totali = len(y_predicted_ok)
# predizioni_giuste = 0
# predizioni_errate = 0
# for i in range(len(y_predicted_ok)):
#     if abs(y_test_true[i] - y_predicted_ok[i]) < 0.5:
#         predizioni_giuste = predizioni_giuste+1
#     else:
#         predizioni_errate = predizioni_errate+1
# print("totali/giuste/errate: ", predizioni_totali, "-", predizioni_giuste, "-", predizioni_errate, "-")
# print(predizioni_giuste/predizioni_totali)

# dizio_val = {"traccia": Data_predicting.metadata["trace_name"], "y_Mano": y_test_true, "y_predict": y_predicted_ok,
#              "delta": delta_y}
# print(len(Data_predicting.metadata["trace_name"]), len(y_test_true), len(y_predicted_ok), len(delta_y))


# datapandas_val = pd.DataFrame.from_dict(dizio_val)
# # TODO cambia nome del file
# if salva_predizioni:
#     datapandas_val.to_csv(pat_tent + str(tentativo) +
#                       nome_predizione + str(tentativo) + ".csv", index=False)
# # TODO cambia se vuoi inserire la predizione nel file di metadata
# # Data_predicting.metadata["Pred_tent_" + str(tentativo)] = y_predicted_ok
# # Data_predicting.crea_custom_dataset(hdf5_predicting, csv_predicting)


