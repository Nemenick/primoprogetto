"""import dask.dataframe as dd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings"""
from Classe_sismogramma_v3 import ClasseDataset

"""
csvin = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/metadata/metadata_Instance_events_10k.csv'
hdf5in = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/data/Instance_events_counts_10k.hdf5'
coltot = ["trace_name", "station_channels", "trace_P_arrival_sample", "trace_polarity",
          "trace_P_uncertainty_s", "source_magnitude", "source_magnitude_type"]
txt_data = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/cattivi_down_1Som/SOM_solo_down_1a_iterazione_data.txt'
txt_metadata = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/cattivi_down_1Som/SOM_solo_down_1a_iterazione_metadata.txt'          
classiup_path = 'C:/Users/GioCar/Desktop/Tesi_5/SOM/3classes_up.txt'
classidown_path = 'C:/Users/GioCar/Desktop/Tesi_5/SOM/4classes_down.txt'
nomi_up = "Selezionati_up.csv"
nomi_down = "Selezionati_down.csv"
"""

# Dataset_d = ClasseDataset()
# Dataset_d.acquisisci_old(hdf5in, csvin, coltot, nomi_down)
# Dataset_d.leggi_classi_txt(classidown_path)
# classidown_buone = []
# vettore_indici = []
# Dataset_d.ricava_indici_classi(classidown_buone, vettore_indici)
# Dataset_d.elimina_tacce(vettore_indici)
# Dataset_d.crea_custom_dataset('C:/Users/GioCar/Desktop/SOM_solo_down_2a_iterazione.hdf5','C:/Users/GioCar/Desktop/SOM_solo_down_2a_iterazione.csv')

# TODO  visualizza classi
"""
hdf5 = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/cattivi_down_1Som/SOM_solo_down_1a_iterazione.hdf5'
csv = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/cattivi_down_1Som/SOM_solo_down_1a_iterazione.csv'
classidown_path = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/cattivi_down_1Som/4classes_down_1a_iterazione.txt'

Dataset = ClasseDataset()
Dataset.leggi_custom_dataset(hdf5, csv)
Dataset.leggi_classi_txt(classidown_path)

semiampiezza_ = 100
classi_indici = [i for i in range(1, 17)]  # TODO da cambiare
cartella = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/cattivi_down_1Som/Immagini_classi'
for i in classi_indici:
    vettore_indici = []
    Dataset.ricava_indici_classi([i], vettore_indici)
    nomepng = "2a_iterazione_classe" + str(i)
    print("classe "+str(i), vettore_indici, [i])
    Dataset.plotta(vettore_indici, semiampiezza_, nomepng, percosro_cartellla=cartella)
"""

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


# classidown_buone = []
# vettore_indici = []
# Dataset_d.ricava_indici_classi(classidown_buone, vettore_indici)
# Dataset_d.elimina_tacce(vettore_indici)
# Dataset.finestra(200)
# Dataset.to_txt(txt_data, txt_metadata)
# print("1",Dataset.sismogramma.shape,"2", len(Dataset.metadata["trace_name"]))

# semiampiezza_ = 100
# print("\n\n\n\n")
# for i in classiup_indici:
#     vettore_indici = []
#     Dataset_u.ricava_indici_classi([i], vettore_indici)
#     nomepng = "classe_up" + str(i)
#     print("up", vettore_indici, [i])
    # Dataset_u.plotta(vettore_indici, semiampiezza_, nomepng)
# for i in classidown_indici:
#     vettore_indici = []
#     Dataset_d.ricava_indici_classi([i], vettore_indici)
#     nomepng = "classe_down" + str(i)
#     Dataset_d.plotta(vettore_indici, semiampiezza_, nomepng)



