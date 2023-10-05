# Faccio predizione con tutti i tentativi More

# nohup /home/silvia/Documents/GitHub/primoprogetto/venv/bin/python /home/silvia/Documents/GitHub/primoprogetto/Codici/Deep_Load_bag.py &> Codici/Tentativi/Bag_predictions/Zoutput.txt
# /home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/More_6/2/Tentativo_More_6_2.hdf5
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

undecidable = False
# houtu = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/Dati_sotto_1_perc/data_up_sotto_1_perc.hdf5'
# coutu = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/Dati_sotto_1_perc/metadata_up_sotto_1_perc.csv'
# # houtd = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/Dati_sotto_1_perc/data_down_sotto_1_perc.hdf5'
# # coutd = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/Dati_sotto_1_perc/metadata_down_sotto_1_perc.csv'
# houtd = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/Dati_sotto_1_perc/data_down_sotto_1_perc.hdf5'
# coutd = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/Dati_sotto_1_perc/metadata_down_sotto_1_perc.csv'

hdf5_predicting = '/home/silvia/Desktop/SCSN(Ross)/Ross_test_polarity_Normalizzate20_New1-1_data.hdf5'
csv_predicting = '/home/silvia/Desktop/SCSN(Ross)/Ross_test_polarity_Normalizzate20_New1-1_metadata.csv'

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

# hdf5_predicting = '/home/silvia/Desktop/SCSN(Ross)/Ross_test_undecidable_Normalizzate20_New1-1_data.hdf5'
# csv_predicting = '/home/silvia/Desktop/SCSN(Ross)/Ross_test_undecidable_Normalizzate20_New1-1_metadata.csv'
# undecidable = True

Data_predicting = ClasseDataset()
Data_predicting.leggi_custom_dataset(hdf5_predicting, csv_predicting)
sample_train = len(Data_predicting.sismogramma)

lung = len(Data_predicting.sismogramma[0])
semi_amp = 80
path_tentativi = f'/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi'
salva_predizioni = True
time_shift = 1
nome_predizione = "_NEW_Predizioni_Bag_Ross_polarity_shift_+1"
batch = 1024

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
# """
xtest = np.array(xtest)

if undecidable:
    y_test_true = ["undecidable" for i in range(len(y_test_true))]


dizio_test = {"traccia": Data_predicting.metadata["trace_name"], "y_Mano": y_test_true}
datapandas_test = pd.DataFrame.from_dict(dizio_test)
if salva_predizioni:
    datapandas_test.to_csv(f"{path_tentativi}/Bag_predictions/{nome_predizione}.csv", index=False)            # TODO ATTENZIONE



# trovo tutti i models (i percorsi dei file hdf5)
l_mods_p = []
a=os.scandir(path_tentativi)
for it in a:
    if it.is_dir():
        c = it.path.split("/") # c = [home, silvia,..., Tentativi, More_#1]
        if c[-1][0] == "M":
            b=os.scandir(it.path)
            for bb in b:        # bb.path = /home/s.../More_#1/#2
                if bb.is_dir():
                    bbb = bb.path.split("/")
                    l_mods_p.append(f"{bb.path}/Tentativo_{bbb[-2]}_{bbb[-1]}.hdf5")
l_mods_p.sort() # [/home/s.../More_#1/#2/Tentativo_More#1_#2, ...]

continua = False
file = open(f"{path_tentativi}/Bag_predictions/DOVE_STO.txt","a")
for i, modello in enumerate(l_mods_p):

    splittato = modello.split("/")
    print(f"{i} {splittato[-3]}_{splittato[-2]}\t{modello}\n")

    if modello == '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/More_6/2/Tentativo_More_6_2.hdf5' or continua:
        continua = True # faccio da More_2_3 fino a fine
        tf.keras.backend.clear_session()                        ## FIXME MOOOOLTA ATTENZIONE mi libera tutta la memoria allocata su GPU  
        model = keras.models.load_model(modello)                ##       (sembra funzionare ma dopo alcuni training ho comunque un crash!)
        model.summary()                                         ##       (Noto che con bathsize = 512 già riempio 80% memoria, come fa a gestire 4096?)

        yp_test = model.predict(xtest, batch_size=batch)
        yp_ok_test = []
        for i in yp_test:
            yp_ok_test.append(i[0])
        yp_ok_test = np.array(yp_ok_test)

        # acc_fin_test = 0
        # for delta in delta_y_test:
        #     if abs(delta) < 0.5:
        #         acc_fin_test += 1
        # acc_fin_test = acc_fin_test / len(delta_y_test)
        

        datapandas_test = pd.DataFrame.from_dict({f"Pred_{splittato[-3]}_{splittato[-2]}" : yp_ok_test})
        if salva_predizioni:
            datapandas_test.to_csv(f"{path_tentativi}/Bag_predictions/{nome_predizione}_Pred_{splittato[-3]}_{splittato[-2]}.csv", index=False)
        file.write(f"Ho finito il training Pred_{splittato[-3]}_{splittato[-2]}\n")
file.close()