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
classiup = 'C:/Users/GioCar/Desktop/Tesi_5/SOM/3classes_up.txt'
classidown = 'C:/Users/GioCar/Desktop/Tesi_5/SOM/4classes_down.txt'
nomi_up = "Selezionati_up.csv"
nomi_down = "Selezionati_down.csv"

Dataset_d = ClasseDataset()
Dataset_u = ClasseDataset()

Dataset_d.acquisisci_old(hdf5in, csvin, coltot, nomi_down)
Dataset_u.acquisisci_old(hdf5in, csvin, coltot, nomi_up)

Dataset_u.leggi_classi_txt(classiup)
Dataset_d.leggi_classi_txt(classidown)


