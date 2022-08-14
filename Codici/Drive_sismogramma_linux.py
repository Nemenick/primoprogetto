"""import dask.dataframe as dd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings"""
from Classe_sismogramma_v3 import ClasseDataset

# TODO seleziona classi
"""
hdf5 = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_up_Velocimeter_4s.hdf5'
csv = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_up_Velocimeter_4s.csv'
classi_path = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_up/classes_up.txt'

Data = ClasseDataset()
Data.leggi_custom_dataset(hdf5, csv)
Data.leggi_classi_txt(classi_path)

classi_buone = [i+1 for i in range(25)]
classi_buone.remove(2)
classi_buone.remove(5)
classi_buone.remove(25)

indici = []
Data.ricava_indici_classi(classi_buone, indici)
Data.elimina_tacce_indici(indici)
hdf5out = '/home/silvia/Desktop/Instance_Data/Tre_4s/Up_1_iterazione/data_clas_2_5_25.hdf5'
csvout = '/home/silvia/Desktop/Instance_Data/Tre_4s/Up_1_iterazione/metadata_clas_2_5_25.csv'
Data.crea_custom_dataset(hdf5out, csvout)

txt_data = '/home/silvia/Desktop/Instance_Data/Tre_4s/Up_1_iterazione/2_5_25/data_up_2_5_25.txt'
txt_metadata = '/home/silvia/Desktop/Instance_Data/Tre_4s/Up_1_iterazione/2_5_25/metadata_up_2_5_25.txt'
Data.to_txt(txt_data, txt_metadata)
"""
# TODO  visualizza classi
# """
hdf5 = '/home/silvia/Desktop/Instance_Data/Tre_4s/Down_1_iterazione/7_9/data_clas_7_9.hdf5'
csv = '/home/silvia/Desktop/Instance_Data/Tre_4s/Down_1_iterazione/7_9/metadata_clas_7_9.csv'
classidown_path = '/home/silvia/Desktop/Instance_Data/Tre_4s/Down_1_iterazione/7_9/7_9_post_7_5x5_classes.txt'

Dataset = ClasseDataset()
Dataset.leggi_custom_dataset(hdf5, csv)
Dataset.leggi_classi_txt(classidown_path)

semiampiezza_ = 100
classi_indici = [i for i in range(1, 26)]  # TODO da cambiare
# TODO crea la cartella Immagini_classi
cartella = '/home/silvia/Desktop/Instance_Data/Tre_4s/Down_1_iterazione/7_9/Immagini_down_7_9'
for i in classi_indici:
    vettore_indici = []
    Dataset.ricava_indici_classi([i], vettore_indici)
    nomepng = "2a_iterazione_classe" + str(i)
    print("classe "+str(i), vettore_indici, [i])
    Dataset.plotta(vettore_indici, semiampiezza_, nomepng, percosro_cartellla=cartella)
# """

# TODO seleziona classi buone (da dataset big posso eliminare up/down in contemporaneo, non creo 2 dataset e poi unisco)
"""
hdf5in_1a_iter = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/cattivi_down_1Som/SOM_solo_down_1a_iterazione.hdf5'
csvin_1a_iter = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/cattivi_down_1Som/SOM_solo_down_1a_iterazione.csv'
hdf5in_ori = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/data/Simple_velocity_down.hdf5'
csvin_ori = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/metadata/Simple_velocity_down.csv'
hdf5out = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/BUONI_DOWN.hdf5'
csvout = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/BUONI_DOWN.csv'
classi_path = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/cattivi_down_1Som/4classes_down_1a_iterazione.txt'

Dataset = ClasseDataset()
Dataset.leggi_custom_dataset(hdf5in_1a_iter, csvin_1a_iter)
Dataset.leggi_classi_txt(classi_path)
classi_buone = [1, 7, 8, 9, 10, 11, 12, 14, 16]
indici_buone = []
Dataset.ricava_indici_classi(classi_buone, vettore_indici=indici_buone)
Dataset.elimina_tacce_indici(indici_buone)   # ho selezionato solo i cattivi della 1a iterazione

nomi_cattivi = Dataset.metadata["trace_name"]
print("\n\nQUI", type(nomi_cattivi), len(nomi_cattivi))

Dataset_ori = ClasseDataset()
Dataset_ori.leggi_custom_dataset(hdf5in_ori, csvin_ori)
Dataset_ori.elimina_tacce_nomi(nomi_cattivi)
Dataset_ori.crea_custom_dataset(hdf5out, csvout)
"""

# TODO genera txt per SOM
"""
hdf5 = '/home/silvia/Desktop/Instance_Data/Tre_4s/Down_1_iterazione/5_21_23/data_clas_5_21_23.hdf5'
csv = '/home/silvia/Desktop/Instance_Data/Tre_4s/Down_1_iterazione/5_21_23/metadata_clas_5_21_23.csv'

txt_data = '/home/silvia/Desktop/Instance_Data/Tre_4s/Down_1_iterazione/5_21_23/data_down_5_21_23.txt'
txt_metadata = '/home/silvia/Desktop/Instance_Data/Tre_4s/Down_1_iterazione/5_21_23/metadata_down_5_21_23.txt'

Dataset = ClasseDataset()
Dataset.leggi_custom_dataset(hdf5, csv)
# Dataset.finestra(200)
Dataset.to_txt(txt_data, txt_metadata)
"""

# Todo Dividui up/down
"""
hdf5 = '/home/silvia/Desktop/Instance_Data/Due_8s/data_selected_Polarity_Velocimeter_8s.hdf5'
csv = '/home/silvia/Desktop/Instance_Data/Due_8s/metadata_Instance_events_selected_Polarity_Velocimeter_8s.csv'
hdf5out = '/home/silvia/Desktop/Instance_Data/Tre_8s/data_down_Velocimeter_8s.hdf5'
csvout = '/home/silvia/Desktop/Instance_Data/Tre_8s/metadata_down_Velocimeter_8s.csv'

Dataset = ClasseDataset()
Dataset.leggi_custom_dataset(hdf5, csv)
elimina = []
for i in range(len(Dataset.sismogramma)):
    if Dataset.metadata["trace_polarity"][i] == "positive":
        elimina.append(i)
Dataset.elimina_tacce_indici(elimina)

Dataset.crea_custom_dataset(hdf5out, csvout)

"""