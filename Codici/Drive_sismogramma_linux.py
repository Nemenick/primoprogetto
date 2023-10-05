# nohup /home/silvia/Documents/GitHub/primoprogetto/venv/bin/python /home/silvia/Documents/GitHub/primoprogetto/Codici/Drive_sismogramma_linux.py &> Codici/Zoutput.txt

# import dask.dataframe as dd
import os
import h5py
import math
import seaborn
import dask.dataframe as dd
import matplotlib.pyplot as plt
# from matplotlib import colors
import obspy
import pandas as pd
# import time
# import warnings
import numpy as np
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from Classe_sismogramma_v3 import ClasseDataset
from tensorflow import keras
from keras.layers import  Dropout
import gc
import shutil

# TODO acquisisci new
""" 
Dati = ClasseDataset()
hdf5 = '/home/silvia/Desktop/Instance_Data/data'
csv = '/home/silvia/Desktop/Instance_Data/metadata_Instance_events_v2.csv'
hdf5_mio = /home/silvia/Desktop/Instance_undecidable_data.hdf5'
csv_mio = /home/silvia/Desktop/Instance_undecidable_metadata.csv'
colonne = ['trace_name','station_code','station_channels','trace_start_time','trace_P_arrival_sample',
'trace_polarity','trace_P_uncertainty_s','source_magnitude','source_magnitude_type','source_origin_time',
'source_latitude_deg','source_longitude_deg','trace_Z_snr_db']
Dati.acquisisci_new(hdf5, csv, colonne)
Dati.finestra(400)
Dati.demean()
Dati.crea_custom_dataset(hdf5_mio, csv_mio)
 """

# TODO ricava longitudine, latitudine metadata
"""
hdf5in = '/home/silvia/Desktop/Instance_Data/data'
csvin = '/home/silvia/Desktop/Instance_Data/metadata_Instance_events_v2.csv'
col_sel = ['trace_name', 'source_latitude_deg', 'source_longitude_deg', 'source_origin_time', 'station_code',
           'station_channels', 'trace_start_time', 'trace_P_arrival_sample',
           'trace_polarity', 'trace_P_uncertainty_s', 'source_magnitude', 'source_magnitude_type', 'trace_Z_snr_db'
           ]

hdf5out = '/home/silvia/Desktop/data_buttare.hdf5'
csvout = '/home/silvia/Desktop/metadata_pol_veloc_more_metadata.csv'

Datain = ClasseDataset()
Datain.acquisisci_new(hdf5in, csvin, col_sel)

Datain.crea_custom_dataset(hdf5out, csvout)
"""

# TODO ricava longitudine, latitudine metadata, SOLO CSV
"""
csvin = '/home/silvia/Desktop/Instance_Data/metadata_Instance_events_v2.csv'
col_sel = ['trace_name', 'source_latitude_deg', 'source_longitude_deg', 'source_origin_time', 'station_code',
           'station_channels', 'trace_start_time', 'trace_P_arrival_sample',
           'trace_polarity', 'trace_P_uncertainty_s', 'source_magnitude', 'source_magnitude_type', 'trace_Z_snr_db'
           ]

hdf5out = '/home/silvia/Desktop/data_buttare.hdf5'
csvout = '/home/silvia/Desktop/ATTENTOmetadata_pol_veloc_more_metadata.csv'

Datain = ClasseDataset()
Datain.acquisisci_new_csv(csvin, col_sel)
print(len(Datain.metadata['trace_name']))
Datain.crea_custom_dataset(hdf5out, csvout)
"""

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
"""
hdf5 = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/Som_up/data_Velocimeter_Buone_up_4s.hdf5'
csv = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/Som_up/metadata_Velocimeter_Buone_up_4s.csv'
classidown_path = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/Som_up/Som_up_buoni_classes.txt'

Dataset = ClasseDataset()
Dataset.leggi_custom_dataset(hdf5, csv)
Dataset.leggi_classi_txt(classidown_path)

semiampiezza_ = 100
classi_indici = [i for i in range(1, 26)]  # TODO da cambiare
# TODO crea la cartella Immagini_classi
cartella = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/Som_up/Immagini_up_buoni'
for i in classi_indici:
    vettore_indici = []
    Dataset.ricava_indici_classi([i], vettore_indici)
    # vettore_indici = vettore_indici[0:200]
    nomepng = "Up_buoni_classe" + str(i)
    print("classe "+str(i), vettore_indici, [i])
    Dataset.plotta(vettore_indici, semiampiezza_, nomepng, percosro_cartellla=cartella)
"""

# TODO seleziona classi buone (da dataset big posso eliminare up/down in contemporaneo, non creo 2 dataset e poi unisco)
"""
hdf5in_ori = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s.hdf5'
csvin_ori = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s.csv'

sto_qui = '/home/silvia/Desktop/Instance_Data/Tre_4s'
hdf_list = ['/Down_1_iterazione/4_8_10/data_clas_4_8_10', '/Down_1_iterazione/5_21_23/data_clas_5_21_23',
            '/Down_1_iterazione/7_9/data_clas_7_9', '/Up_1_iterazione/4/data_clas_4',
            '/Up_1_iterazione/24/data_clas_24', '/Up_1_iterazione/2_5_25/data_clas_2_5_25']
csv_list = ['/Down_1_iterazione/4_8_10/metadata_clas_4_8_10', '/Down_1_iterazione/5_21_23/metadata_clas_5_21_23',
            '/Down_1_iterazione/7_9/metadata_clas_7_9', '/Up_1_iterazione/4/metadata_clas_4',
            '/Up_1_iterazione/24/metadata_clas_24', '/Up_1_iterazione/2_5_25/metadata_clas_2_5_25']
classi_list = ['/Down_1_iterazione/4_8_10/4_8_10_post_5_classes', '/Down_1_iterazione/5_21_23/5_21_23_post_7_classes',
               '/Down_1_iterazione/7_9/7_9_post_7_classes', '/Up_1_iterazione/4/4_post_10_classes',
               '/Up_1_iterazione/24/24_post_10_classes', '/Up_1_iterazione/2_5_25/2_5_25_post_7_classes']


for i in range(6):
    hdf_list[i] = sto_qui + hdf_list[i] + '.hdf5'
    csv_list[i] = sto_qui + csv_list[i] + '.csv'
    classi_list[i] = sto_qui + classi_list[i] + '.txt'


hdf5out = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s_.hdf5'
csvout = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s_.csv'

classi_buone = [[19, 25, 20],
                [11, 18, 20, 21, 15, 16, 23, 19, 5, 24],
                [5, 9, 12, 15, 10, 18, 20, 1, 6, 11, 16, 17, 21, 22, 23, 24, 25],
                [10, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25],
                [3, 4, 13, 14, 18, 1, 5, 8, 9, 10, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25],
                [2, 4, 9, 14, 18, 24, 5, 10, 11, 19, 21, 22, 25]]

nomi_cattivi = []
for i in range(6):
    Dataset = ClasseDataset()
    Dataset.leggi_custom_dataset(hdf_list[i], csv_list[i])
    Dataset.leggi_classi_txt(classi_list[i])
    indici_buone = []
    Dataset.ricava_indici_classi(classi_buone[i], vettore_indici=indici_buone)
    Dataset.elimina_tacce_indici(indici_buone)   # ho selezionato solo i cattivi della 1a iterazione
    nomi_cattivi = nomi_cattivi + list(Dataset.metadata["trace_name"])  
    print("elimino tracce in numero ", len(Dataset.metadata["trace_name"]), "da", classi_list[i])
    # print("QUI", type(nomi_cattivi), len(nomi_cattivi))

# Dataset_ori = ClasseDataset()
# Dataset_ori.leggi_custom_dataset(hdf5in_ori, csvin_ori)
# Dataset_ori.elimina_tacce_nomi(nomi_cattivi)
# Dataset_ori.crea_custom_dataset(hdf5out, csvout)

"""

# TODO genera txt per SOM
"""
# hdf5 = '/home/silvia/Desktop/Instance_Data/Tre_4s/Down_1_iterazione/5_21_23/data_clas_5_21_23.hdf5'
# csv = '/home/silvia/Desktop/Instance_Data/Tre_4s/Down_1_iterazione/5_21_23/metadata_clas_5_21_23.csv'
#
# txt_data = '/home/silvia/Desktop/Instance_Data/Tre_4s/Down_1_iterazione/5_21_23/data_down_5_21_23.txt'
# txt_metadata = '/home/silvia/Desktop/Instance_Data/Tre_4s/Down_1_iterazione/5_21_23/metadata_down_5_21_23.txt'
hdf5 = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_data_100Hz_sbagliati27.hdf5'
csv = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_metadata_100Hz_sbagliati27.csv'

txt_data = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_data_100Hz_sbagliati27.txt'
txt_metadata = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_metadata_100Hz_sbagliati27.txt'

Dataset = ClasseDataset()
Dataset.leggi_custom_dataset(hdf5, csv)
# elimina = []
# print(len(Dataset.sismogramma), len(elimina))
# for i in range(len(Dataset.sismogramma)):
#     if Dataset.metadata["trace_Z_snr_db"][i] >= 10:
#         elimina.append(i)
# Dataset.elimina_tacce_indici(elimina)
# # Dataset.crea_custom_dataset('/home/silvia/Desktop/Instance_Data/Tre_4s/data_down_Velocimeter_4s_SRN_L_10.hdf5',
'/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_down_Velocimeter_4s_SRN_L_10.csv')
# print(len(Dataset.sismogramma), len(elimina))
# Dataset.finestra(200)
Dataset.to_txt(txt_data, txt_metadata)
"""

# TODO Dividi dataset up/down o altro
"""
# hdf5 = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_data_100Hz_sbagliati27.hdf5'
# csv = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_metadata_100Hz_sbagliati27.csv'
#
# hdf5out = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_data_100Hz_sbagliati27_Up.hdf5'
# csvout = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_metadata_100Hz_sbagliati27_Up.csv'

hdf5in = '/home/silvia/Desktop/SCSN(Ross)/Ross_test_polarity_Normalizzate20_New1-1_data.hdf5'
csvin = '/home/silvia/Desktop/SCSN(Ross)/Ross_test_polarity_Normalizzate20_New1-1_metadata.csv'

hdf5out = '/home/silvia/Desktop/SCSN(Ross)/Ross_test_polarity_Normalizzate20_New1-1_SNR_L_data.hdf5'
csvout = '/home/silvia/Desktop/SCSN(Ross)/Ross_test_polarity_Normalizzate20_New1-1_SNR_L_metadata.csv'

# txt_data = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_data_100Hz_sbagliati27_Up.txt'
# txt_metadata = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_metadata_100Hz_sbagliati27_Up.txt'

Dataset = ClasseDataset()
Dataset.leggi_custom_dataset(hdf5in, csvin)
print(len(Dataset.sismogramma), len(Dataset.metadata["trace_name"]))
seleziona = []
for i in range(len(Dataset.sismogramma)):
    if Dataset.metadata["trace_Z_snr_db"][i] < 10:
        seleziona.append(i)
        # print(i)
Dataset = Dataset.seleziona_indici(seleziona)
# Dataset.finestra(200)
print(len(Dataset.sismogramma), len(Dataset.metadata["trace_polarity"]))
Dataset.crea_custom_dataset(hdf5out, csvout)
# Dataset.to_txt(txt_data, txt_metadata)
"""

# TODO visualizza
"""
hdf5 = '/home/silvia/Desktop/Sample_dataset/_Mio/Instance_events_gm_10k_mio.hdf5'
csv = '/home/silvia/Desktop/Sample_dataset/_Mio/metadata_Instance_events_10k.csv'

Dataset = ClasseDataset()
Dataset.leggi_custom_dataset(hdf5, csv)
semiampiezza_ = 220
cartella = '/home/silvia/Desktop/'

# vettore_indici = [62, 632, 7299, 1022, 9037] vettore_indici = [8] (200)
vettore_indici = [8]

nomepng = "_Presentazione_Undecidable"
Dataset.plotta(vettore_indici,  namepng=nomepng, semiampiezza=semiampiezza_, percosro_cartellla=cartella)
"""

# TODO genera Custom Normalizzato
"""
Dati = ClasseDataset()

csvin = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s.csv'
hdf5in = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s.hdf5'

csvout = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s_Normalizzate.csv'
hdf5out = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s_Normalizzate.hdf5'

Dati.leggi_custom_dataset(hdf5in, csvin)  # Leggo il dataset
Dati.normalizza()
Dati.crea_custom_dataset(hdf5out, csvout)
"""

# TODO Grafico Instance Data (mappa)
"""
# hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Cattive/data_Velocimeter_Cattive_4s.hdf5'
# csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Cattive/metadata_Velocimeter_Cattive_4s.csv'

csvin = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s_Normalizzate_New1-1.csv'
hdf5in = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s_Normalizzate_New1-1.hdf5'


Data = ClasseDataset()
Data.leggi_custom_dataset(hdf5in, csvin)

min_lat = 35.1392   # np.min(Data.metadata['source_latitude_deg'])
max_lat = 48.166    # np.max(Data.metadata['source_latitude_deg'])
min_lon = 5.3923    # np.min(Data.metadata['source_longitude_deg'])
max_lon = 18.9612   # np.max(Data.metadata['source_longitude_deg'])
print(min_lat, max_lat, min_lon, max_lon)
fig, grafico = plt.subplots()  # figsize=(25, 20)
m = Basemap(llcrnrlon=min_lon,  urcrnrlon=max_lon, llcrnrlat=min_lat, urcrnrlat=max_lat, resolution='i')
m.drawcoastlines()
m.fillcontinents()

m.drawparallels(np.arange(36, 52, 2), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(6, 22, 2), labels=[1, 1, 0, 1])
m.drawcountries()
plt.hist2d(x=Data.metadata['source_longitude_deg'],
           y=Data.metadata['source_latitude_deg'],
           bins=(208, 200),
           cmap='inferno',
           zorder=1,
           alpha=0.99,
           norm=colors.LogNorm()
           )

# hb = grafico.hexbin(x=Data.metadata['source_longitude_deg'],
#                     y=Data.metadata['source_latitude_deg'],
#                     gridsize=200,
#                     cmap='inferno',
#                     bins="log",
#                     zorder=1,
#                     )
# cb = fig.colorbar(hb, ax=grafico)
# cb.set_label('counts')
plt.colorbar()
e_t = [43, 45, 9.5, 11.8]
e_v = [37.5, 38.5, 14.5, 16]

x_t = [9.5, 9.5, 11.8, 11.8, 9.5]
y_t = [43, 45, 45, 43, 43]
x_v, y_v = [14.5, 14.5, 16, 16, 14.5], [37.5, 38.5, 38.5, 37.5, 37.5]
plt.plot(x_t, y_t, zorder=2, linewidth=1, color="deeppink")
plt.plot(x_v, y_v, zorder=2, linewidth=1, color="orange")
plt.title("Number of events in Dataset A")
plt.savefig('/home/silvia/Desktop/Italia_Tracce_DPI_', dpi=300)
plt.show()
"""

# TODO seleziona tracce (devi avere un modo per ricavare indici)
"""
# hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s.hdf5'
# csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s.csv'
#
# hdf5out = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s_primi100.hdf5'  # TODO
# csvout = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s_primi100.csv'


Datain = ClasseDataset()
Datain.leggi_custom_dataset(hdf5in, csvin)

path = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi'
tentativi = "27"
predizioni = pd.read_csv(path + '/' + tentativi + '/Predizioni_Pollino_tentativo_' + tentativi + '.csv')

vettore_indici = []  # TODO
for i in range(len(predizioni["delta"])):
    if predizioni["delta"][i] >= 0.5:
        vettore_indici.append(i)

Dataout = Datain.seleziona_indici(vettore_indici)
print(Dataout.metadata["centrato"], "\n##########", Dataout.centrato)
Dataout.crea_custom_dataset(hdf5out, csvout)

# vettore_verita = []
# for i in range(len(Dataout.sismogramma)):
#     vettore_verita.append((Dataout.sismogramma[i] == Datain.sismogramma[vettore_indici[i]]).all())
# print(np.array(vettore_verita).all())
# lista_nomi = Datain.metadata["trace_name"][0:100]
# print(np.array(Dataout.metadata["trace_name"]) == np.array(lista_nomi)).all()
"""

# TODO conta longitudine latitudine
"""
hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s_Normalizzate.hdf5'
csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s_Normalizzate.csv'
Data = ClasseDataset()
Data.leggi_custom_dataset(hdf5in, csvin)

cont_test = 0
cont_val = 0
cont_pol = 0

e_pol = [39.7, 40.2, 15.8, 16.4]            # Dati presenti nella zona del pollino
e_test = [43, 45, 9.5, 11.8]
e_val = [37.5, 38.5, 14.5, 16]

for i in range(len(Data.sismogramma)):
    if e_test[0] < Data.metadata['source_latitude_deg'][i] < e_test[1] and e_test[2] \
            < Data.metadata['source_longitude_deg'][i] < e_test[3]:
        cont_test = cont_test + 1
    if e_val[0] < Data.metadata['source_latitude_deg'][i] < e_val[1] and e_val[2] \
            < Data.metadata['source_longitude_deg'][i] < e_val[3]:
        cont_val = cont_val + 1
    if e_pol[0] < Data.metadata['source_latitude_deg'][i] < e_pol[1] and e_pol[2] \
            < Data.metadata['source_longitude_deg'][i] < e_pol[3]:
        cont_pol = cont_pol + 1

print("test = ", cont_test, " val = ", cont_val, "pol = ", cont_pol)
"""

# TOdo verifica pollino in insta
"""
hdf5ins = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_lon_lat_time_4s.hdf5'
csvins = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_lon_lat_time_4s.csv'

hdf5pol = '/home/silvia/Desktop/Pollino_All/Pollino_All_data_100Hz.hdf5'
csvpol = '/home/silvia/Desktop/Pollino_All/Pollino_All_metadata_100Hz.csv'

Datains, Datapol = ClasseDataset(), ClasseDataset()
Datains.leggi_custom_dataset(hdf5ins, csvins)
Datapol.leggi_custom_dataset(hdf5pol, csvpol)

tracce_pol_in_inst = []
tempi_ins = []
for j in range(len(Datains.metadata["source_origin_time"])):
    tempi_ins.append(obspy.UTCDateTime(Datains.metadata["source_origin_time"][j]))

for i in range(len(Datapol.metadata["source_origin_time"])):
    tempo_pol = obspy.UTCDateTime(Datapol.metadata["source_origin_time"][i])
    for j in range(len(Datains.metadata["source_origin_time"])):
        if tempi_ins[j] - 1 < tempo_pol < tempi_ins[j] + 1:
            tracce_pol_in_inst.append(i)
            break
print(len(tracce_pol_in_inst), tracce_pol_in_inst)
"""

# TODO ricava instance in pollino e viceversa
"""
hdf5ins = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s_Normalizzate.hdf5'
csvins = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_lon_lat_time_4s.csv'

hdf5ins_out = 
'/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s_Normalizzate_SENZA_POLLINO.hdf5'
csvins_out = 
'/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s_Normalizzate_SENZA_POLLINO.csv'

hdf5pol = '/home/silvia/Desktop/Pollino_All/Pollino_All_data_100Hz.hdf5'
csvpol = '/home/silvia/Desktop/Pollino_All/Pollino_All_metadata_100Hz.csv'

hdf5pol_out = '/home/silvia/Desktop/Pollino_All/Pollino_data_comuni_inst.hdf5'
csvpol_out = '/home/silvia/Desktop/Pollino_All/Pollino_metadata_comuni_ins.csv'

Datains, Datapol = ClasseDataset(), ClasseDataset()
Datains.leggi_custom_dataset(hdf5ins, csvins)
Datapol.leggi_custom_dataset(hdf5pol, csvpol)


## INST IN POLLINO E POLLINOININST VEDI UN FILE DI TESTO da qualche parte
# Data_pol_in_inst = Datapol.seleziona_indici(pollino_in_inst)
# Data_inst_in_pol = Datains.seleziona_indici(inst_in_pollino)
# Data_pol_in_inst.crea_custom_dataset(hdf5pol_out, csvpol_out)
# Data_inst_in_pol.crea_custom_dataset(hdf5ins_out, csvins_out)
# Datains.elimina_tacce_indici(inst_in_pollino)
# Datains.crea_custom_dataset(hdf5ins_out, csvins_out)
"""

# TODO Confronta rete
"""
# SGD con Momentum, momentum = 0.6    18
# SGD con Momentum, momentum = 0.9    20
# Adam con epsilon = 1e-05            21
# Adam con epsilon = 1e-03            22
# Adam con epsilon = 1e-01            23
# labels = ["SGD, m=0.6", "SGD, m=0.9", "ADAM, ε=1e-05", "ADAM, ε=1e-03", "ADAM, ε=1e-01"]
labels = ["SGD m=0.75", "ADAM ε=1e-3"]
path = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi'
tent_buoni = ['28', '27']
colori_train = ["royalblue", "red"]
colori_val = ["dodgerblue", "orangered"]
le = len(tent_buoni)
Storie = [{} for i in range(le)]
for i in range(le):
    Storie[i] = pd.read_csv(path+'/'+tent_buoni[i]+'/Storia_train_'+tent_buoni[i]+'.csv')

fig, graf = plt.subplots()
for i in range(le):
    plt.plot(Storie[i]["loss_train"], label="train_Loss_" + labels[i], color=colori_train[i])
    plt.plot(Storie[i]["loss_val"], label="val_Loss_"+labels[i], color=colori_val[i])
    # plt.yscale("log")
    graf.set_ylim(0.0245, 0.15)
    # graf.set_xlim(-2, 100)
plt.legend()
plt.title("Loss")
plt.savefig(path+'/_Loss ADAM vs SGD')
plt.show()

fig, graf = plt.subplots()
for i in range(le):
    plt.plot(Storie[i]["acc_train"], label="train_Acc_" + labels[i], color=colori_train[i])
    plt.plot(Storie[i]["acc_val"], label="val_Acc_" + labels[i], color=colori_val[i])
    graf.set_ylim(0.96, 0.996)
plt.legend()
plt.title("Accuracy")
plt.savefig(path+'/_Accuracy ADAM vs SGD')
plt.show()

min_los = [np.min(Storie[i]["loss_val"]) for i in range(le)]
print(min_los)
max_ac = [np.max(Storie[i]["acc_val"]) for i in range(le)]
print(max_ac)
"""

# TODO matrice confusione Predizione TEST
"""
# "y_Mano_test", "y_predict_test"
fig_name = ['Hara_tentativo_55']
titoli = ["Western Japan test set"]                                  # TODO cambia
path = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi'
ipath = '/home/silvia/Desktop/'
predizione = '/Predizioni_Hara_tentativo_'  # '/Predizioni_test_tentativo_'
tentativi = ['55']

le = len(tentativi)

# predizioni["y_Mano_test"][j] == 1 and predizioni["delta_test"][j] < 0.5:
# predizioni["y_Mano_pol"][j] == 1 and predizioni["delta_val"][j] < 0.5:
for i in range(le):
    tp, tn, fp, fn = 0, 0, 0, 0
    # predizioni = pd.read_csv(path + '/' + tentativi[i] + predizione + tentativi[i] + '.csv') # TODO
    predizioni = pd.read_csv(path + '/' + tentativi[i] + predizione + tentativi[i] + '.csv')
    for j in range(len(predizioni["traccia"])):
        # predizioni["y_Mano_test"][j] == 1 and predizioni["delta_test"][j] < 0.5:          # TODO
        # predizioni["y_Mano_pol"][j] == 1 and predizioni["delta_val"][j] < 0.5:

        if predizioni["y_Mano"][j] == 1 and predizioni["delta"][j] < 0.5:
            tp += 1
        if predizioni["y_Mano"][j] == 0 and predizioni["delta"][j] < 0.5:
            tn += 1
        if predizioni["y_Mano"][j] == 1 and predizioni["delta"][j] >= 0.5:
            fn += 1
        if predizioni["y_Mano"][j] == 0 and predizioni["delta"][j] >= 0.5:
            fp += 1
    print("SONO TUTTI ? (tutti), tp+tn+..", len(predizioni["y_predict"]), tp+tn+fp+fn)
    print(tp, fn, "\n", fp, tn)
    print((tp+tn) / (fp+fn+tn+tp))
    df = pd.DataFrame([[tp, fn], [fp, tn]], columns=["Positive", "Negative"])

    # plot a heatmap with annotation
    ax = seaborn.heatmap(df, annot=True, fmt=".7g", annot_kws={"size": 20}, cmap="Blues", cbar=False)
    plt.xlabel("Predicted polarity (network)", fontsize=23, labelpad=33)
    plt.ylabel("Assigned polarity (catalogue)", fontsize=23, labelpad=33)
    plt.title(titoli[i], fontsize=23, pad=30)
    ax.set_yticklabels(['Positive', 'Negative'], fontsize=15)
    ax.set_xticklabels(['Positive', 'Negative'], fontsize=15)
    ax.text(-0.1, 1.35,  r'$\mathbf{B}$', transform=ax.transAxes, fontsize=30,
            verticalalignment='top')
    plt.savefig(ipath+'/Confusion_matrix_'+fig_name[i]+".jpg", bbox_inches='tight', dpi=300)
    plt.show()
"""

# TODO istogrammi vari
"""
# istogramma magnitudo
# hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s.hdf5'
# csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s.csv'

hdf5in = '/home/silvia/Desktop/Pollino_All/Pollino_All_data_100Hz.hdf5'
csvin = '/home/silvia/Desktop/Pollino_All/Pollino_All_metadata_100Hz.csv'

Data = ClasseDataset()
Data.leggi_custom_dataset(hdf5in, csvin)

magnitudini = []
for mag in Data.metadata['source_magnitude']:
    if mag != 'unknown':                        # circa 7 non avevano magnitudo
        magnitudini.append(float(mag))
    
q = [0.25, 0.5, 0.75]
q = np.quantile(magnitudini, q)
fig, ax = plt.subplots()
plt.yscale('log')
ax.hist(magnitudini, edgecolor="black", bins=26)
plt.axvline(x=q[0], c="orange", ls="--", lw="2")
plt.axvline(x=q[1], c="r", ls="--", lw="2")
plt.axvline(x=q[2], c="orange", ls="--", lw="2")
plt.xlabel("Magnitudo")
plt.ylabel("Numero di eventi")
plt.title("Magnitudo Dataset_2")
plt.savefig('/home/silvia/Desktop/Magnitudo_Pollino')
plt.show()
"""
"""
# Istogramma tempi
hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s.hdf5'
csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s.csv'
Data = ClasseDataset()
Data.leggi_custom_dataset(hdf5in, csvin)

lat_min_amatrice, lat_max_amatrice = 42.47, 43.07
lon_min_amatrice, lon_max_amatrice = 13.04, 13.29

tempi = []
year_tempi = []
for j in range(len(Data.sismogramma)):
    tempi.append(obspy.UTCDateTime(Data.metadata["source_origin_time"][j]))
    year_tempi.append(tempi[j].year)

q = [0.25, 0.5, 0.75]
q = np.quantile(year_tempi, q)


fig, ax = plt.subplots()
# plt.yscale('log')
ax.hist(year_tempi, edgecolor="black", bins=15)
plt.axvline(x=q[0], c="orange", ls="--", lw="2")
plt.axvline(x=q[1], c="r", ls="--", lw="2")
plt.axvline(x=q[2], c="orange", ls="--", lw="2")
plt.xlabel("Anno")
plt.ylabel("Numero di eventi")
plt.show()
"""
"""

csvin = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s_Normalizzate_New1-1.csv'
hdf5in = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s_Normalizzate_New1-1.hdf5'
classi_path = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_up/classes_up.txt'

Data = ClasseDataset()
Data.leggi_custom_dataset(hdf5in, csvin)
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
# TODO predict tracce mislabeled (analisi som, heterogeneous)
"""
csvin = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s_Normalizzate_New1-1.csv'
hdf5in = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s_Normalizzate_New1-1.hdf5'
houtu = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/data_up_sotto_1_perc.hdf5'
coutu = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/metadata_up_sotto_1_perc.csv'
houtd = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/data_down_sotto_1_perc.hdf5'
coutd = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/metadata_down_sotto_1_perc.csv'
Data = ClasseDataset()
Data.leggi_custom_dataset(hdf5in,csvin)

selezionau = []
selezionad = []

# Data.leggi_classi_txt('/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/NEWclasses.txt')
for i in range(len(Data.sismogramma)):
    if Data.metadata["trace_polarity"][i] == 'positive':
        selezionau.append(i)
    else:
        selezionad.append(i)
        # print(i)
Datau = Data.seleziona_indici(selezionau)
Datad = Data.seleziona_indici(selezionad)

classi = []
with open('/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/NEWclasses.txt', 'r') as f:
    for line in f:
        if line:  # avoid blank lines
            classi.append(int(float(line.strip())))
Datau.classi = classi[0:len(Datau.sismogramma)]
Datad.classi = classi[len(Datau.sismogramma):]
# print(len(Datad.sismogramma), len(Datau.sismogramma), len(Datau.classi), len(Datad.classi))

classi_up_l_1_1 = [18, 33, 41, 49]
classi_up_l_1_2 = [34]
classi_down_l_1_1 = [24, 30, 31, 39, 40, 46]
classi_down_l_1_2 = [47, 54]

cl_up_indici, cl_down_indici = [], []
Datau.ricava_indici_classi(classi_up_l_1_1, cl_up_indici)
Datad.ricava_indici_classi(classi_down_l_1_1, cl_down_indici)
Datau_1 = Datau.seleziona_indici(cl_up_indici)
Datad_1 = Datad.seleziona_indici(cl_down_indici)

cl_up_indici, cl_down_indici = [], []
Datau.ricava_indici_classi(classi_up_l_1_2, cl_up_indici)
Datad.ricava_indici_classi(classi_down_l_1_2, cl_down_indici)
Datau_2 = Datau.seleziona_indici(cl_up_indici)
Datad_2 = Datad.seleziona_indici(cl_down_indici)

Datau.sismogramma = np.concatenate((Datau_1.sismogramma, Datau_2.sismogramma))
for key in Datau.metadata:
    Datau.metadata[key] = Datau_1.metadata[key] + Datau_2.metadata[key]
Datad.sismogramma = np.concatenate((Datad_1.sismogramma, Datad_2.sismogramma))
for key in Datad.metadata:
    Datad.metadata[key] = Datad_1.metadata[key] + Datad_2.metadata[key]
# print(len(Datad.sismogramma), len(Datau.sismogramma))

Datau.crea_custom_dataset(houtu,coutu)
Datad.crea_custom_dataset(houtd,coutd)
"""

# TODO Affidabilità predizioni
"""
# keys(INSTANCE_Test) = ['traccia_test', 'y_Mano_test', 'y_predict_test', 'delta_test']
tentativo = '52'
nbin = '15'
percorso = "/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/" + tentativo
datd = pd.read_csv(percorso + "/Predizioni_Roos_Normalizzate20_Testset_tentativo_" + tentativo + ".csv",
                   dtype={"y_Mano": float, "y_predict": float, 'delta': float})
print(datd.keys())

predizioni = [[0 for _ in range(4)] for __ in range(nbin)]
# predizioni [5][0] = pred totali intervallo [0.5,0.6]
# predizioni [5][1] = pred giuste intervallo [0.5,0.6]
# predizioni [5][2] = pred errate intervallo [0.5,0.6]
# predizioni [5][3] = pred giuste/totali intervallo [0.5,0.6]
for i in range(1, len(datd)):
    indice = math.floor(float(datd['y_predict'][i]) * nbin)
    indice = 19 if indice == 20 else indice
    predizioni[indice][0] += 1
    if datd['delta'][i] < 0.5:
        predizioni[indice][1] += 1
    else:
        predizioni[indice][2] += 1
for i in range(len(predizioni)):
    predizioni[i][3] = predizioni[i][1] / (float(predizioni[i][0]) + 0.000001)
dizio_pred = pd.DataFrame(predizioni, columns=["Totali", "Giuste", "Errate", "Affidabilità"])
dizio_pred.to_excel(percorso + "/Affidabilità_test_Ross_tentativo_" + tentativo + ".xlsx")
"""

# TODO reshape for Phasenet
"""
percorsohdf5 = '/home/silvia/Desktop/Sample_dataset/data/Instance_events_counts_10k.hdf5'
hdf5out = '/home/silvia/Desktop/Sample_dataset/data/Instance_events_counts_10k_reshaped_PhaseNet.hdf5'
percorsocsv = '/home/silvia/Desktop/Sample_dataset/metadata/metadata_Instance_events_10k.csv'
colonne = ['trace_name','station_code','station_channels','trace_start_time','trace_P_arrival_sample',
'trace_polarity','trace_P_uncertainty_s','source_magnitude','source_magnitude_type','source_origin_time',
'source_latitude_deg','source_longitude_deg']

filehdf5 = h5py.File(percorsohdf5, 'r')
dataset = filehdf5.get("data")


datd = pd.read_csv(percorsocsv, usecols=colonne, engine="python", on_bad_lines="skip",
                           dtype={'source_latitude_deg': 'object',
                                  'source_longitude_deg': 'object',
                                  'source_origin_time': 'object'})
allmetadata = {}
for i in colonne:  # genera metadata["colname"] = np.array["colname"]
    allmetadata[i] = np.array(datd[i])                                    # LEGGO CSV

for key in allmetadata:
    print(key, allmetadata[key])
nomidata = allmetadata["trace_name"]                     # Presi dal file CSV
# print(nomidata)
# print(type(nomidata), nomidata)
Data_shape_2 = []
Data_shape = []
index = 0
for i in nomidata:
    Data_shape.append(list(dataset.get(i)))
Data_shape = np.array(Data_shape)
for i in range(len(Data_shape)):
    arrivo = allmetadata["trace_P_arrival_sample"][i]
    Data_shape_2.append([])
    for j in range(3):
        Data_shape_2[i].append(Data_shape[-1][j][arrivo - 1399:arrivo + 1602])  # fai -1399, + 1602
        # Data_shape[-1][j] = Data_shape[-1][j][arrivo-1499:arrivo+1499]
Data_shape_2 = np.array(Data_shape_2)
print(Data_shape_2.shape)


lung = len(Data_shape_2[0][0])

print("######################", lung)
print(Data_shape_2.shape)

# Data_shape[1].reshape(3,12000)
Data_reshaped = Data_shape_2[0].reshape(lung, 3)

filehdf5out = h5py.File(hdf5out, 'w')
filehdf5out.create_group("data")
data = filehdf5out.get("data")
for i in range(len(Data_shape_2)):  # len(Data_shape)
    Data_reshaped = np.float32(Data_shape_2[i].reshape(lung, 3))
    data.create_dataset(name=nomidata[i], data=Data_reshaped)
    if i % 100 == 0:
        print(i)

print(Data_shape_2.shape)
print(type(Data_shape_2[1]))
print(Data_reshaped.shape)
print(type(Data_reshaped), Data_reshaped.shape)
print("vedi qui", type(Data_shape_2[1][1][1]), type(Data_reshaped[1][1]))

filehdf5out.close()
filehdf5.close()
"""

# TODO confronto timeshift plot
"""
pat = 'Predizioni_shift/'
file = [open(pat+'predizioni_shift_Ross_tent_39.txt', "r")]
numero_confronto = len(file)
predizioni = [[[] for j in range(3)] for i in range(numero_confronto)]
# file = [open(pat+"predizioni_shift_Instance_tent_52.txt", "r"), open(pat+"predizioni_shift_Instance_tent_53.txt", "r"),
#         open(pat+"predizioni_shift_Instance_tent_66.txt", "r"), open(pat+"predizioni_shift_Instance_tent_79.txt", "r"),
#         open(pat+"predizioni_shift_Instance_tent_39.txt", "r") ]

path_save = '/home/silvia/Desktop'
name_save = "Timeshift_39_Ross"
labels = ["39", "53", "66", "79", "39(old)"]
# labels = ["Timeshift in training set", "NO Timeshift in training set"]
colori = ["blue", "red", "orange", "green","purple"]

for k in range(numero_confronto):
    for line in file[k]:
        a = line.split()
        # print(a)
        predizioni[k][0].append(int(a[0]))
        for i in range(1, 3):
            predizioni[k][i].append(float(a[i]))


print(predizioni)
fig, axs = plt.subplots(1, 2)
fig.set_figheight(5)
fig.set_figwidth(10)
for k in range(numero_confronto):
    axs[0].plot(predizioni[k][0], predizioni[k][1], label=labels[k], color=colori[k])  # TODO plt.
    axs[0].legend()
    axs[0].set_title('Test Loss')

    axs[1].plot(predizioni[k][0], predizioni[k][2], label=labels[k], color=colori[k])

axs[1].set_title('Test Accuracy')
axs[1].axhline(0.5, color='k', ls='dashed', lw=1)
axs[1].axhline(0.75, color='k', ls='dashed', lw=1)
axs[1].legend()

for ax in axs.flat:
    ax.set(xlabel='T  (translation samples in test set)')


axs[1].axvline(linestyle="--", color="red", linewidth="0.5")
plt.savefig(path_save + "/" + name_save, dpi=300)
"""

# TODO data from Hara
"""
pat = '/home/silvia/Desktop/Hara/Validation'
hdf = '/home/silvia/Desktop/Hara/Validation/Hara_validation_data_Normalizzate_1-1.hdf5'
csv = '/home/silvia/Desktop/Hara/Validation/Hara_validation_metadata_Normalizzate_1-1.csv'
polarity = 'validation_100Hz_polarity.npy'
data = 'validation_100Hz_waveform.npy'

a = np.load(pat + "/" + data)
b = np.load(pat + "/" + polarity)

Hara = ClasseDataset()
Hara.sismogramma = a
Hara.centrato = True
Hara.metadata["trace_name"] = ["trace_"+str(i) for i in range(len(a))]
Hara.metadata["trace_polarity"] = ["positive" if b[i][0] == 1 else "negative" for i in range(len(a))]
Hara.demean()
# Hara.crea_custom_dataset(hdf, csv)
visualizzare = [i for i in range(20)]
# Hara.plotta(visualizzare, namepng="Hara_figure", percosro_cartellla=pat)
Hara.normalizza()
Hara.crea_custom_dataset(hdf, csv)
Hara.plotta(visualizzare, namepng="Hara_figure_normalizzate", percosro_cartellla=pat)
"""

# TODO ricava indici test e classi (Som_updown_secondo) (ERRATO)
"""
hdf5in = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s_Normalizzate_New1-1.hdf5'
csvin = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s_Normalizzate_New1-1.csv'

hdfout = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/Test.hdf5'
csvout = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/Test.csv'
classi_out = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/NEWclasses_test.txt'
indici_out = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/indici_test.txt'
dati = ClasseDataset()
dati.leggi_custom_dataset(hdf5in, csvin)
dati.leggi_classi_txt('/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/NEWclasses.txt')

estremi_test = [43, 45, 9.5, 11.8]
indici_test = []

for k in range(len(dati.sismogramma)):
    if estremi_test[0] < dati.metadata['source_latitude_deg'][k] < estremi_test[1] and estremi_test[2] \
            < dati.metadata['source_longitude_deg'][k] < estremi_test[3]:
        indici_test.append(k)  # li farò eliminare dal trainset
dati_test = dati.seleziona_indici(indici_test)
dati_test.crea_custom_dataset(hdfout,csvout)
classi_test = pd.DataFrame.from_dict({"classi_test": dati_test.classi},dtype=int)
classi_test.to_csv(classi_out,index=False)

indici_test = pd.DataFrame.from_dict({"indici_test": indici_test},dtype=int)
indici_test.to_csv(indici_out,index=False)
"""

# TODO ricava indici test e classi METODO GIUSTO (Som_updown_secondo)
"""
hdf5inup = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_up_Velocimeter_4s.hdf5'
csvinup = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_up_Velocimeter_4s.csv'

hdf5indown = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_down_Velocimeter_4s.hdf5'
csvindown = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_down_Velocimeter_4s.csv'

hdfout = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/Test.hdf5'
csvout = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/Test.csv'
classi_out = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/NEWclasses_test.txt'
indici_out = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/indici_test_ordine_up_down.txt'

dati_all = ClasseDataset()
dati_up = ClasseDataset()
dati_down = ClasseDataset()
dati_up.leggi_custom_dataset(hdf5inup, csvinup)
dati_down.leggi_custom_dataset(hdf5indown, csvindown)

dati_all.leggi_classi_txt('/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/NEWclasses.txt')

estremi_test = [43, 45, 9.5, 11.8]
indici_test_ordine_up_down = []

for k in range(len(dati_up.sismogramma)):
    if estremi_test[0] < dati_up.metadata['source_latitude_deg'][k] < estremi_test[1] and estremi_test[2] \
            < dati_up.metadata['source_longitude_deg'][k] < estremi_test[3]:
        indici_test_ordine_up_down.append(k+1)
a = len(dati_up.sismogramma)
for k in range(len(dati_down.sismogramma)):
    if estremi_test[0] < dati_down.metadata['source_latitude_deg'][k] < estremi_test[1] and estremi_test[2] \
            < dati_down.metadata['source_longitude_deg'][k] < estremi_test[3]:
        indici_test_ordine_up_down.append(k+a+1)

indici_test = pd.DataFrame.from_dict({"indici_test": indici_test_ordine_up_down},dtype=int)
indici_test.to_csv(indici_out,index=False, header= False)

######################################### AlTRa sezione#################################àà
predizioni_test = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/52/Predizioni_test_tentativo_52.csv'
up_errate_pat = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/indici_up_errate.txt'
down_errate_pat = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/indici_down_errate.txt'
pred_up, pred_down = pd.read_csv(predizioni_test), pd.read_csv(predizioni_test)

pred_up = pred_up.loc[pred_up["y_Mano_test"] == 1 ]
pred_down = pred_down.loc[pred_down["y_Mano_test"] == 0 ]

pred_up.index = [i+1 for i in range(len(pred_up.index))]
pred_down.index = [i+1 + len(pred_up.index) for i in range(len(pred_down.index))]
# STO QUI ora ho gli indici del test in ordine. nel senso da 1 a 5567 up il resto down. incrocio con il file
# indici_test_ordine_up_down.txt per capire quale è la classe di appartenenza

pred_up_errate = pred_up.loc[pred_up["delta_test"] >= 0.5 ]
pred_down_errate = pred_down.loc[pred_down["delta_test"] >= 0.5 ]
# FIXME a questo punto ho gli indici delle tracce misclassified, partendo da 1!
# se trovo 330 in pred_up_errate.index significa che la traccia 330-esima delle tracce del test ordinate prima up e poi down ha predizione errata
# se trovo 6087 in pred_down_errate.index significa che la traccia 6087-esima delle tracce del test ordinate prima up e poi down ha predizione errata
# controllo incrociato con indici_test_ordine_up_down mi darà gli indici sbagliati!!!
# indici_test_ordine_up_down[pred_up.index] mi restituisce gli indici delle tracce up del test che la rete sbaglia
# (indici rispetto tutte le 161198 tracce ordinate prima up poi down)

pred_up_errate_indici = pd.DataFrame.from_dict({"indici_test_errati_up": pred_up_errate.index},dtype=int)
pred_up_errate_indici.to_csv(up_errate_pat, index=False, header=False)

pred_down_errate_indici = pd.DataFrame.from_dict({"indici_test_errati_down": pred_down_errate.index},dtype=int)
pred_down_errate_indici.to_csv(down_errate_pat, index=False, header= False)
"""
"""
hdf5in= '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s.hdf5'
csvin = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s.csv'

pred = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/52/Predizioni_test_tentativo_52.csv'

txt_u = "/home/silvia/Desktop/SOM/SOM_errori/Tentativo_52/Instance_test/Instance_up_test_errate"
txt_d = "/home/silvia/Desktop/SOM/SOM_errori/Tentativo_52/Instance_test/Instance_down_test_errate"

Dati = ClasseDataset()
Dati.leggi_custom_dataset(hdf5in, csvin)

estremi_test = [43, 45, 9.5, 11.8]
indici_test=[]
for k in range(len(Dati.sismogramma)):
    if estremi_test[0] < Dati.metadata['source_latitude_deg'][k] < estremi_test[1] and estremi_test[2] \
            < Dati.metadata['source_longitude_deg'][k] < estremi_test[3]:
        indici_test.append(k)

Dati = Dati.seleziona_indici(indici_test)

prediz = pd.read_csv(pred)
up_sbagliate = []
down_sbagliate = []

for i in range(len(Dati.sismogramma)):
    if prediz["delta_test"][i] >= 0.5:
        if prediz.y_Mano_test[i] == 0:
            down_sbagliate.append(i)
        else:
            up_sbagliate.append(i)

Data_u_err = Dati.seleziona_indici(up_sbagliate)
Data_d_err = Dati.seleziona_indici(down_sbagliate)
Data_u_err.to_txt(txt_u)
Data_d_err.to_txt(txt_d)
print(len(Data_u_err.sismogramma))
print(len(Data_d_err.sismogramma))
"""

# TODO Metti insime predizioni bag
"""
import os
list_csv = []
pat = f'/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/Bag_predictions/'
a=os.scandir(pat)
for it in a:
    if it.path[-1] == "v":
        list_csv.append(it.path)
list_csv.sort()

datapandas1 = pd.read_csv(list_csv[0])
for i in list_csv[1:]:
    datapandas2 = pd.read_csv(i)
    for j in list(datapandas2.columns):
        datapandas1[j] = datapandas2[j]
datapandas1.to_csv("/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/Bag_predictions/File_buono.csv", index= False)
"""
"""
# Metti insieme predizioni versione 2
import pandas as pd

pred = ["/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/More_1/0/Predizioni_Instance_Undecidable_More_1_0.csv",
 "/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/More_1/1/Predizioni_Instance_Undecidable_More_1_1.csv",
 "/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/More_1/2/Predizioni_Instance_Undecidable_More_1_2.csv",
 "/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/More_1/3/Predizioni_Instance_Undecidable_More_1_3.csv",
 "/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/More_1/4/Predizioni_Instance_Undecidable_More_1_4.csv",
 "/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/More_1/5/Predizioni_Instance_Undecidable_More_1_5.csv",
 "/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/More_1/6/Predizioni_Instance_Undecidable_More_1_6.csv",
 ]
predizioni = [0 for i in range(7)]

for i, p in enumerate(pred):
    predizioni[i] = pd.read_csv(p)
for i, p in enumerate(predizioni):
    p[f"y_predict_{i}"] =  p[f"y_predict"] 
newdataframe = pd.DataFrame.from_dict(predizioni[0]["traccia"])
for i, p in enumerate(predizioni):
    newdataframe[f"y_predict_{i}"] =p[f"y_predict_{i}"] 
newdataframe["y_mano"] = "unndecidable"
newdataframe.to_csv("/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/More_1/Predizioni_Instance_undecidable.csv")
newdataframe"""

# TODO predizioni 70 reti ROSS
"""
import pandas as pd
import numpy as np
pat_und = "/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/Bag_predictions/Predizioni_ross_undecidable_shift+1_more1_to_more10/File_buono.csv"
pat_pol = "/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/Bag_predictions/Predizioni_ross_polarity_shift+1_more1_to_more10/File_buono.csv"
datapd_pol = pd.read_csv(pat_pol)
datapd_und = pd.read_csv(pat_und)
mean_pol = datapd_pol.loc[:,list(datapd_pol.columns)[2:]].mean(axis=1)  # elimino col traccia e y_mano!
mean_und = datapd_und.mean(axis=1)

lis = []
for i, col in enumerate(datapd_pol.columns):
    if i > 1:
        lis.append(col)

df_bool_pol = (datapd_pol.loc[:,lis] >=0.5)
votation_pol = df_bool_pol.sum(axis=1)

lis = []
for i, col in enumerate(datapd_und.columns):
    if i > 1:
        lis.append(col)

df_bool_und = (datapd_und.loc[:,lis] >=0.5)
votation_und = df_bool_und.sum(axis=1)

binnaggio = 100
import matplotlib.pyplot as plt
fig, axi = plt.subplots(2,2, figsize=(13,7.5))
mean_pol.hist(bins=binnaggio, edgecolor="black", ax=axi[0][0], weights=np.ones(len(mean_pol)) / len(mean_pol))
mean_pol.hist(bins=40, edgecolor="black", ax=axi[0][1], weights=np.ones(len(mean_pol)) / len(mean_pol))
mean_und.hist(bins=binnaggio, edgecolor="black",   ax=axi[1][0],weights=np.ones(len(mean_und)) / len(mean_und))
mean_und.hist(bins=40, edgecolor="black",   ax=axi[1][1],weights=np.ones(len(mean_und)) / len(mean_und))
"""
