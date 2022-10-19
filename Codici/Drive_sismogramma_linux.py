# import dask.dataframe as dd
# import h5py
import matplotlib.pyplot as plt
from matplotlib import colors
# import pandas as pd
# import time
# import warnings
import numpy as np
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

hdf5out = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s.hdf5'
csvout = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s.csv'

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

# Todo Dividui dataset up/down o altro
"""
hdf5 = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s.hdf5'
csv = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s.csv'

hdf5out = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_down_Velocimeter_4s_copia.hdf5'
csvout = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_down_Velocimeter_4s.csv'

txt_data = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_down_4s.txt'
txt_metadata = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_down_4s.txt'

Dataset = ClasseDataset()
Dataset.leggi_custom_dataset(hdf5, csv)
elimina = []
for i in range(len(Dataset.sismogramma)):
    if Dataset.metadata["trace_polarity"][i] != 'negative':
        elimina.append(i)
Dataset.elimina_tacce_indici(elimina)

Dataset.crea_custom_dataset(hdf5out, csvout)
# Dataset.to_txt(txt_data, txt_metadata)

"""

# TODO visualizza
"""
hdf5 = '/home/silvia/Desktop/Pollino/Pollino_100Hz_data.hdf5'
csv = '/home/silvia/Desktop/Pollino/Pollino_100Hz_metadata.csv'

Dataset = ClasseDataset()
Dataset.leggi_custom_dataset(hdf5, csv)
semiampiezza_ = 100

cartella = '/home/silvia/Desktop/Pollino'

vettore_indici = range(len(Dataset.sismogramma))

nomepng = 'Pollino_figure_100Hz'
Dataset.plotta(vettore_indici, semiampiezza_, nomepng, percosro_cartellla=cartella)
"""

# TODO genera Custom Normalizzato
"""
Dati = ClasseDataset()

csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s.csv'
hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s.hdf5'

csvout = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s_Normalizzate.csv'
hdf5out = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s_Normalizzate.hdf5'

Dati.leggi_custom_dataset(hdf5in, csvin)  # Leggo il dataset
Dati.normalizza()
Dati.crea_custom_dataset(hdf5out, csvout)
"""

# TODO ricava longitudine, latitudine metadata
"""
hdf5in = '/home/silvia/Desktop/Instance_Data/data'
csvin = '/home/silvia/Desktop/Instance_Data/metadata_Instance_events_v2.csv'
col_sel = ['trace_name', 'source_latitude_deg', 'source_longitude_deg', 'source_origin_time', 'station_code',
           'station_channels', 'trace_start_time', 'trace_P_arrival_sample',
           'trace_polarity', 'trace_P_uncertainty_s', 'source_magnitude', 'source_magnitude_type'
           ]

hdf5out = '/home/silvia/Desktop/data_buttare.hdf5'
csvout = '/home/silvia/Desktop/metadata_pol_veloc_more_metadata.csv'

Datain = ClasseDataset()
Datain.acquisisci_new(hdf5in, csvin, col_sel)

Datain.crea_custom_dataset(hdf5out, csvout)
"""

# TODO Grafico Instance Data
"""
hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s.hdf5'
csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s.csv'

Data = ClasseDataset()
Data.leggi_custom_dataset(hdf5in, csvin)
figura, grafico_lon = plt.subplots()
hb = grafico_lon.hexbin(x=Data.metadata['source_longitude_deg'],
                        y=Data.metadata['source_latitude_deg'],
                        gridsize=200,
                        cmap='inferno',
                        bins="log")
min_lat = np.min(Data.metadata['source_latitude_deg'])
max_lat = np.max(Data.metadata['source_latitude_deg'])
min_lon = np.min(Data.metadata['source_longitude_deg'])
max_lon = np.max(Data.metadata['source_longitude_deg'])
grafico_lon.axis([min_lon, max_lon, min_lat, max_lat])
grafico_lon.set_title("Hexagon binning")
grafico_lon.set_title("Hexagon binning")
cb = figura.colorbar(hb, ax=grafico_lon)
cb.set_label('counts')
# plt.axvline(min_lat, c='navy')
print(min_lon, max_lon)
print(min_lat, max_lat)
# plt.colorbar()
plt.show()
"""

# TODO seleziona tracce (devi avere un modo per ricavare indici)
"""
hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s.hdf5'
csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s.csv'

hdf5out = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s_primi100.hdf5'  # TODO
csvout = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s_primi100.csv'

vettore_indici = [] # TODO

Datain = ClasseDataset()
Datain.leggi_custom_dataset(hdf5in, csvin)
Dataout = Datain.seleziona_indici(vettore_indici)
Dataout.crea_custom_dataset(hdf5out, csvout)

vettore_verita = []
for i in range(len(Dataout.sismogramma)):
    vettore_verita.append((Dataout.sismogramma[i] == Datain.sismogramma[vettore_indici[i]]).all())
print(np.array(vettore_verita).all())
lista_nomi = Datain.metadata["trace_name"][0:100]
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

e_test = [43, 45, 9.5, 11.8]
e_val = [37.5, 38.5, 14.5, 16]

for i in range(len(Data.sismogramma)):
    if e_test[0] < Data.metadata['source_latitude_deg'][i] < e_test[1] and e_test[2] \
            < Data.metadata['source_longitude_deg'][i] < e_test[3]:
        cont_test = cont_test + 1
    if e_val[0] < Data.metadata['source_latitude_deg'][i] < e_val[1] and e_val[2] \
            < Data.metadata['source_longitude_deg'][i] < e_val[3]:
        cont_val = cont_val + 1

print("test = ", cont_test, " val = ", cont_val)

"""


