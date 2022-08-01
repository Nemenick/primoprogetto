"""import dask.dataframe as dd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings"""
from Classe_sismogramma_v3 import ClasseDataset


csvin = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/metadata/metadata_Instance_events_10k.csv'
hdf5in = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/data/Instance_events_counts_10k.hdf5'
coltot = ["trace_name", "station_channels", "trace_P_arrival_sample", "trace_polarity",
          "trace_P_uncertainty_s", "source_magnitude", "source_magnitude_type"]
classiup_path = 'C:/Users/GioCar/Desktop/Tesi_5/SOM/3classes_up.txt'
classidown_path = 'C:/Users/GioCar/Desktop/Tesi_5/SOM/4classes_down.txt'
nomi_up = "Selezionati_up.csv"
nomi_down = "Selezionati_down.csv"

Dataset_d = ClasseDataset()
Dataset_u = ClasseDataset()

Dataset_d.acquisisci_old(hdf5in, csvin, coltot, nomi_down)
Dataset_u.acquisisci_old(hdf5in, csvin, coltot, nomi_up)

Dataset_u.leggi_classi_txt(classiup_path)
print(Dataset_u.classi)
Dataset_d.leggi_classi_txt(classidown_path)




# som down controlla [12, 17, 18, 22]
classiup_indici = [19, 20, 15]
# som up controlla [2, 17, 22, 23]
classidown_indici = [20, 15]

semiampiezza_ = 100
print("\n\n\n\n")
for i in classiup_indici:
    vettore_indici = []
    Dataset_u.ricava_indici_classi([i], vettore_indici)
    nomepng = "classe_up" + str(i)
    print("up", vettore_indici, [i])
    Dataset_u.plotta(vettore_indici, semiampiezza_, nomepng)
for i in classidown_indici:
    vettore_indici = []
    Dataset_d.ricava_indici_classi([i], vettore_indici)
    nomepng = "classe_down" + str(i)
    Dataset_d.plotta(vettore_indici, semiampiezza_, nomepng)



