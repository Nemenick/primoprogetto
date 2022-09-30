"""import dask.dataframe as dd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings"""
from Classe_sismogramma_v3 import ClasseDataset


# TODO  visualizza classi
"""
hdf5 = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/Buoni_Down_simple/BUONI_DOWN.hdf5'
csv = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/Buoni_Down_simple/BUONI_DOWN.csv'
classidown_path = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/Buoni_Down_simple/classes_Buoni_down.txt'

Dataset = ClasseDataset()
Dataset.leggi_custom_dataset(hdf5, csv)
Dataset.leggi_classi_txt(classidown_path)

semiampiezza_ = 100
classi_indici = [i for i in range(1, 26)]  # TODO da cambiare
# TODO crea la cartella Immagini_classi
cartella = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/Buoni_Down_simple/Immagini_classi'
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

# TODO genera txt per SOM
"""
hdf5 = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/Buoni_Down_simple/BUONI_DOWN.hdf5'
csv = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/Buoni_Down_simple/BUONI_DOWN.csv'
txt_data = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/Buoni_Down_simple/BUONI_DOWN_data.txt'
txt_metadata = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/Buoni_Down_simple/BUONI_DOWN_metadata.txt'

Dataset = ClasseDataset()
Dataset.leggi_custom_dataset(hdf5, csv)
Dataset.to_txt(txt_data, txt_metadata)
"""

# TODO  normalizza

hdf5 = "C:/Users/GioCar/Desktop/Tesi_5/data_Velocimeter_Buone_4s.hdf5"
csv = "C:/Users/GioCar/Desktop/Tesi_5/metadata_Velocimeter_Buone_4s.csv"

hdf5out = "C:/Users/GioCar/Desktop/Tesi_5/data_Velocimeter_Buone_normalizzate1_4s.hdf5"
csvout = "C:/Users/GioCar/Desktop/Tesi_5/metadata_Velocimeter_Buone_normalizzate1_4s.csv"

Data = ClasseDataset()
Data.leggi_custom_dataset(hdf5, csv)
Data.normalizza()
Data.crea_custom_dataset(hdf5out, csvout)
