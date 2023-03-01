# import dask.dataframe as dd
import h5py
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


# TODO acquisisci new
"""
Dati = ClasseDataset()
hdf5 = '/home/silvia/Desktop/Sample_dataset/data/Instance_events_gm_10k.hdf5'
hdf5_mio = '/home/silvia/Desktop/Sample_dataset/data/Instance_events_gm_10k_mio.hdf5'
csv = '/home/silvia/Desktop/Sample_dataset/metadata/metadata_Instance_events_10k.csv'
csv_mio = '/home/silvia/Desktop/Sample_dataset/metadata/metadata_Instance_events_10k_mio.csv'
# colonne = ["trace_name", "station_channels", "trace_P_arrival_sample", "trace_polarity",
#            "trace_P_uncertainty_s", "source_magnitude", "source_magnitude_type"] ! RICVEDI
Dati.acquisisci_new(hdf5, csv, colonne)
Dati.finestra(400)
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

# Todo Dividi dataset up/down o altro
"""
hdf5 = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_data_100Hz_sbagliati27.hdf5'
csv = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_metadata_100Hz_sbagliati27.csv'

hdf5out = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_data_100Hz_sbagliati27_Up.hdf5'
csvout = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_metadata_100Hz_sbagliati27_Up.csv'


txt_data = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_data_100Hz_sbagliati27_Up.txt'
txt_metadata = '/home/silvia/Desktop/SOM_Pollino/Pollino_All_metadata_100Hz_sbagliati27_Up.txt'

Dataset = ClasseDataset()
Dataset.leggi_custom_dataset(hdf5, csv)
print(len(Dataset.sismogramma), len(Dataset.metadata["trace_name"]))
elimina = []
for i in range(len(Dataset.sismogramma)):
    if Dataset.metadata["trace_polarity"][i] != 'positive':
        elimina.append(i)
        # print(i)
Dataset.elimina_tacce_indici(elimina)
# Dataset.finestra(200)
Dataset.crea_custom_dataset(hdf5out, csvout)
print(len(Dataset.sismogramma), len(Dataset.metadata["trace_polarity"]))
Dataset.to_txt(txt_data, txt_metadata)
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

hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s.hdf5'
csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s.csv'


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

m.drawparallels(np.arange(36, 52, 2), labels=[1, 1, 0, 1])
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
plt.title("Number of events in Dataset_1")
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

vettore_indici = []  # TODO

Datain = ClasseDataset()
Datain.leggi_custom_dataset(hdf5in, csvin)

path = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi'
tentativi = "27"
predizioni = pd.read_csv(path + '/' + tentativi + '/Predizioni_Pollino_tentativo_' + tentativi + '.csv')

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
titoli = ["_POLLINO_SGD m=0.75", "_POLLINO_ADAM ε=1e-3"]                                  # TODO cambia
path = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi'
tentativi = ['28_1', '27_1']

le = len(tentativi)

# predizioni["y_Mano_test"][j] == 1 and predizioni["delta_test"][j] < 0.5:
# predizioni["y_Mano_pol"][j] == 1 and predizioni["delta_val"][j] < 0.5:
for i in range(le):
    # Create a dataset
    tp, tn, fp, fn = 0, 0, 0, 0
    # predizioni = pd.read_csv(path + '/' + tentativi[i] + '/Predizioni_test_tentativo_' + tentativi[i] + '.csv') # TODO
    predizioni = pd.read_csv(path + '/' + tentativi[i] + '/Predizioni_Pollino_tentativo_' + tentativi[i] + '.csv')
    for j in range(len(predizioni["traccia_val"])):
        # predizioni["y_Mano_test"][j] == 1 and predizioni["delta_test"][j] < 0.5:          # TODO
        # predizioni["y_Mano_pol"][j] == 1 and predizioni["delta_val"][j] < 0.5:

        if predizioni["y_Mano_pol"][j] == 1 and predizioni["delta_val"][j] < 0.5:
            tp += 1
        if predizioni["y_Mano_pol"][j] == 0 and predizioni["delta_val"][j] < 0.5:
            tn += 1
        if predizioni["y_Mano_pol"][j] == 1 and predizioni["delta_val"][j] > 0.5:
            fn += 1
        if predizioni["y_Mano_pol"][j] == 0 and predizioni["delta_val"][j] > 0.5:
            fp += 1
    print("SONO TUTTI ? (tutti), tp+tn+..", len(predizioni["y_pol_predict"]), tp+tn+fp+fn)
    print(tp, fn, "\n", fp, tn)
    df = pd.DataFrame([[tp, fn], [fp, tn]], columns=["Up", "Down"])

    # plot a heatmap with annotation
    seaborn.heatmap(df, annot=True, fmt=".5g", annot_kws={"size": 15}, cmap="Blues", cbar=False)
    plt.xlabel("Predizioni della rete")
    plt.ylabel("Polarità assegnata")
    plt.title(titoli[i])
    plt.savefig(path+'/_Confusion_matrix_'+titoli[i]+".png")
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
# TODO qualcosa tracce mislabeled
"""
csv_up = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_up_Velocimeter_4s.csv'
hdf5_up = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_up_Velocimeter_4s.hdf5'

csv_do = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_down_Velocimeter_4s.csv'
hdf5_do = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_down_Velocimeter_4s.hdf5'

percorsoclassi = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/NEWclasses.txt'
Du, Dd = ClasseDataset(), ClasseDataset()
Du.leggi_custom_dataset(hdf5_up, csv_up)
Dd.leggi_custom_dataset(hdf5_do, csv_do)
print(len(Du.sismogramma), len(Dd.sismogramma))

classi = []
with open(percorsoclassi, 'r') as f:
    for line in f:
        if line:  # avoid blank lines
            classi.append(int(float(line.strip())))
Du.classi = classi[0:len(Du.sismogramma)]
Dd.classi = classi[len(Du.sismogramma):]

classi_up_l_1 = [18, 19, 33, 41, 49]
classi_down_l_1 = [24, 30, 31, 39, 40, 46, 52, 53]
cl_up_indici, cl_down_indici = [], []
Du.ricava_indici_classi(classi_up_l_1, cl_up_indici)
Dd.ricava_indici_classi(classi_down_l_1, cl_down_indici)

Du = Du.seleziona_indici(cl_up_indici)
print(len(cl_down_indici), len(cl_up_indici), cl_down_indici[-1], cl_up_indici[-1])
Dd = Dd.seleziona_indici(cl_down_indici)

Du.crea_custom_dataset('/home/silvia/Desktop/data_U.hdf5', '/home/silvia/Desktop/metadata_U.csv')
Dd.crea_custom_dataset('/home/silvia/Desktop/data_D.hdf5', '/home/silvia/Desktop/metadata_D.csv')
"""

Du, Dd = ClasseDataset(), ClasseDataset()
Du.leggi_custom_dataset('/home/silvia/Desktop/data_U.hdf5', '/home/silvia/Desktop/metadata_U.csv')
Dd.leggi_custom_dataset('/home/silvia/Desktop/data_D.hdf5', '/home/silvia/Desktop/metadata_D.csv')
# Du.plotta(range(len(Du.sismogramma)), 120, "up_dove_up_l_1_perc", '/home/silvia/Desktop')
Dd.plotta(range(len(Dd.sismogramma)), 120, "down_dove_down_l_1_perc", '/home/silvia/Desktop')
# Data.leggi_custom_dataset(hdf5, csv)
# Data.elimina_tacce_indici([133532])
# Data.crea_custom_dataset(hdf5out,csvout)
# 133532

