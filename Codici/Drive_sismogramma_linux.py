# import dask.dataframe as dd
# import h5py
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

min_lat = np.min(Data.metadata['source_latitude_deg'])
max_lat = np.max(Data.metadata['source_latitude_deg'])
min_lon = np.min(Data.metadata['source_longitude_deg'])
max_lon = np.max(Data.metadata['source_longitude_deg'])

fig, grafico = plt.subplots()  # figsize=(25, 20)
m = Basemap(llcrnrlon=min_lon,  urcrnrlon=max_lon, llcrnrlat=min_lat, urcrnrlat=max_lat, resolution='i')
m.drawcoastlines()
m.fillcontinents()

m.drawparallels(np.arange(36, 52, 2), labels=[1, 1, 0, 1])
m.drawmeridians(np.arange(6, 22, 2), labels=[1, 1, 0, 1])
m.drawcountries()
plt.hist2d(x=Data.metadata['source_longitude_deg'],
           y=Data.metadata['source_latitude_deg'],
           bins=(200, 200),
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
plt.plot(x_t, y_t, zorder=2, linewidth=2, color="deeppink")
plt.plot(x_v, y_v, zorder=2, linewidth=2, color="orange")
plt.show()
# plt.savefig('/home/silvia/Desktop/Italia_Bella2')
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

# TOdo verifica pollino
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
hdf5ins = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_lon_lat_time_4s.hdf5'
csvins = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_lon_lat_time_4s.csv'

hdf5ins_out = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_ins_comuni_pollino_4s.hdf5'
csvins_out = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_ins_comuni_pollino_4s.csv'

hdf5pol = '/home/silvia/Desktop/Pollino_All/Pollino_All_data_100Hz.hdf5'
csvpol = '/home/silvia/Desktop/Pollino_All/Pollino_All_metadata_100Hz.csv'

hdf5pol_out = '/home/silvia/Desktop/Pollino_All/Pollino_data_comuni_inst.hdf5'
csvpol_out = '/home/silvia/Desktop/Pollino_All/Pollino_metadata_comuni_ins.csv'

Datains, Datapol = ClasseDataset(), ClasseDataset()
Datains.leggi_custom_dataset(hdf5ins, csvins)
Datapol.leggi_custom_dataset(hdf5pol, csvpol)


inst_in_pollino = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 42, 43, 44, 45, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 278, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 350, 351, 352, 353, 354, 355, 356, 357, 362, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 434, 435, 436, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 518, 519, 520, 521, 522, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 668, 669, 670, 671, 672, 683, 696, 697, 698, 703, 704, 705, 706, 707, 711, 712, 713, 714, 715, 716, 717, 724, 725, 726, 727, 728, 729, 730, 731, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 781, 785, 786, 787, 788, 789, 796, 801, 802, 803, 806, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 846, 847, 848, 849, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 947, 948, 949, 950, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 1011, 1012, 1013, 1014, 1156, 1157, 1158, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1265, 1266, 1267, 1268, 1269, 1277, 1278, 1279, 1280, 1281, 1282]
pollino_in_inst = [134, 135, 136, 137, 138, 162, 163, 164, 165, 166, 167, 168, 218, 219, 220, 221, 321, 322, 323, 324, 325, 326, 327, 328, 329, 333, 334, 335, 336, 337, 361, 362, 363, 400, 401, 402, 403, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 442, 443, 444, 445, 446, 447, 448, 449, 483, 484, 485, 486, 489, 490, 499, 500, 501, 502, 518, 519, 520, 521, 568, 569, 570, 571, 572, 573, 574, 575, 576, 589, 590, 591, 592, 593, 594, 598, 599, 605, 606, 607, 608, 609, 628, 629, 630, 631, 632, 633, 634, 635, 652, 653, 654, 665, 666, 667, 668, 669, 701, 702, 703, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 770, 771, 772, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 885, 886, 887, 888, 889, 890, 891, 892, 893, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 927, 928, 929, 930, 931, 932, 938, 939, 940, 941, 942, 943, 961, 962, 963, 964, 965, 966, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1189, 1190, 1191, 1192, 1193, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1262, 1263, 1264, 1265, 1266, 1267, 1278, 1279, 1280, 1281, 1282, 1283, 1290, 1291, 1292, 1293, 1294, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1325, 1326, 1327, 1328, 1329, 1330, 1381, 1382, 1383, 1384, 1385, 1386, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1445, 1446, 1447, 1448, 1449, 1456, 1457, 1458, 1459, 1460, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1492, 1493, 1494, 1495, 1496, 1497, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1743, 1744, 1745, 1746, 1747, 1748, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1888, 1889, 1890, 1891, 1892, 1893, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 2020, 2021, 2022, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2158, 2159, 2160, 2161, 2162, 2169, 2170, 2171, 2172, 2173, 2189, 2190, 2191, 2192, 2207, 2208, 2209, 2210, 2211, 2212, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2276, 2277, 2278, 2279, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2313, 2363, 2364, 2365, 2368, 2369, 2370, 2382, 2383, 2384, 2385, 2391, 2392, 2393, 2394, 2403, 2404, 2405, 2406, 2424, 2425, 2426, 2427, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2442, 2443, 2444, 2445, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2502, 2503, 2504, 2505, 2506, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2579, 2580, 2581, 2582, 2589, 2590, 2591, 2592, 2593, 2619, 2620, 2621, 2622, 2626, 2627, 2628, 2659, 2660, 2661, 2662, 2663, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2757, 2758, 2759, 2768, 2769, 2770, 2773, 2774, 2775, 2776, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843, 2844, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 2913, 2914, 2921, 2922, 2923, 2924, 2925, 2926, 2927, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2953, 2954, 2955, 2956, 2957, 2958, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2984, 2985, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3046, 3047, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3170, 3171, 3172, 3173, 3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3213, 3214, 3215, 3216, 3264, 3265, 3266, 3267, 3268, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3374, 3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3427, 3428, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3488, 3490, 3491, 3492, 3493, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3524, 3525, 3526, 3527, 3528, 3529, 3530, 3531, 3532, 3544, 3545, 3546, 3547, 3548, 3549, 3582, 3583, 3584, 3585, 3586, 3587, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3608, 3609, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3638, 3639, 3640, 3696, 3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710, 3711, 3712, 3713, 3732, 3733, 3744, 3745, 3746, 3747, 3748, 3749, 3750, 3751, 3752, 3768, 3769, 3770, 3771, 3782, 3783, 3784, 3785, 3786, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808, 3809, 3829, 3830, 3831, 3832, 3833, 3834, 3835, 3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3876, 3877, 3878, 3879, 3880, 3881, 3882, 3883, 3884, 3885, 3886, 3887, 3888, 3889, 3890, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924, 3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961, 3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023, 4024, 4025, 4032, 4033, 4034, 4035, 4046, 4047, 4048, 4049]

Data_pol_in_inst = Datapol.seleziona_indici(pollino_in_inst)
Data_inst_in_pol = Datains.seleziona_indici(inst_in_pollino)

Data_pol_in_inst.crea_custom_dataset(hdf5pol_out, csvpol_out)
Data_inst_in_pol.crea_custom_dataset(hdf5ins_out, csvins_out)
"""

# TODO scegli miglior rete
"""
path = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi'
tent_buoni = ['18', '19', '20', '21', '22', '23']
le = len(tent_buoni)
Storie = [{} for i in range(le)]
for i in range(le):
    Storie[i] = pd.read_csv(path+'/'+tent_buoni[i]+'_ok/'+'Storia_train_'+tent_buoni[i]+'.csv')

fig, graf = plt.subplots()
for i in range(le):
    plt.plot(Storie[i]["loss_val"], label='tentativo_'+tent_buoni[i])
    graf.set_ylim(0.02, 0.15)
    graf.set_xlim(-2, 100)
plt.legend()
plt.title("Loss nei vari train")
# plt.savefig(path+'/Loss_train_buoni')
plt.show()

# fig, graf = plt.subplots()
# for i in range(le):
#
#     plt.plot(Storie[i]["acc_val"], label='tentativo_'+tent_buoni[i])
#     graf.set_ylim(0.97, 0.995)
# plt.legend()
# plt.title("Accuracy nei vari train")
# plt.savefig(path+'/Acc_train_buoni')
# plt.show()

min_los = [np.min(Storie[i]["loss_val"]) for i in range(le)]
print(min_los)
max_ac = [np.max(Storie[i]["acc_val"]) for i in range(le)]
print(max_ac)

"""

# TODO istogrammi vari
"""
hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s.hdf5'
csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s.csv'
Data = ClasseDataset()
Data.leggi_custom_dataset(hdf5in, csvin)
magnitudini = []
for mag in Data.metadata['source_magnitude']:
    magnitudini.append(float(mag))
fig, ax = plt.subplots()
plt.yscale('log')
ax.hist(magnitudini, edgecolor="black", bins=13)
plt.show()
# plt.savefig('/home/silvia/Desktop/Magnitudo')
"""
