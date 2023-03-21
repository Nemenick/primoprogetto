# TODO RIVEDI ERRORI INDICI CLASSI PERCENTUALI SOTTO 1%
"""
# csv_up = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_up_Velocimeter_4s.csv'
# hdf5_up = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_up_Velocimeter_4s.hdf5'
#
# csv_do = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_down_Velocimeter_4s.csv'
# hdf5_do = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_down_Velocimeter_4s.hdf5'
#
# percorsoclassi = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo/NEWclasses.txt'
# Du, Dd = ClasseDataset(), ClasseDataset()
# Du.leggi_custom_dataset(hdf5_up, csv_up)
# Dd.leggi_custom_dataset(hdf5_do, csv_do)
# print(len(Du.sismogramma), len(Dd.sismogramma))
#
# classi = []
# with open(percorsoclassi, 'r') as f:
#     for line in f:
#         if line:  # avoid blank lines
#             classi.append(int(float(line.strip())))
# Du.classi = classi[0:len(Du.sismogramma)]
# Dd.classi = classi[len(Du.sismogramma):]
#
# classi_up_l_1 = [18, 19, 33, 41, 49]
# classi_down_l_1 = [24, 30, 31, 39, 40, 46, 52, 53]
# cl_up_indici, cl_down_indici = [], []
#
# classi_prova = []
# indici_19 = []
# for i in range(len(Du.classi)):
#     for j in classi_up_l_1:
#         if Du.classi[i] == j:
#             classi_prova.append(i)
#             if j == 19:
#                 indici_19.append(len(classi_prova))
# indici_19 = np.array(indici_19)
# print(list(indici_19+1))
# print(len(indici_19))
#
# Du.ricava_indici_classi(classi_up_l_1, cl_up_indici)
# # Dd.ricava_indici_classi(classi_down_l_1, cl_down_indici)
# #
# Du = Du.seleziona_indici(cl_up_indici)
# # print(len(cl_down_indici), len(cl_up_indici), cl_down_indici[-1], cl_up_indici[-1])
# # Dd = Dd.seleziona_indici(cl_down_indici)
# print(Du.classi)
# Du.plotta(np.array(indici_52_53)-1,namepng="52_53", percosro_cartellla='/home/silvia/Desktop')



#
# Du, Dd = ClasseDataset(), ClasseDataset()
# Du.leggi_custom_dataset('/home/silvia/Desktop/data_U_class34.hdf5', '/home/silvia/Desktop/metadata_U_class34.csv')
# Dd.leggi_custom_dataset('/home/silvia/Desktop/data_D_class47_54.hdf5', '/home/silvia/Desktop/metadata_D_class47_54.csv')
#
# Du.plotta(range(len(Du.sismogramma)), 125, 'up_dove_up_l_1_perc_class34', '/home/silvia/Desktop')
# Dd.plotta(range(len(Dd.sismogramma)), 125, 'down_dove_down_l_1_perc_class_47_54', '/home/silvia/Desktop')

# Du.crea_custom_dataset('/home/silvia/Desktop/data_U_class34.hdf5', '/home/silvia/Desktop/metadata_U_class34.csv')
# Dd.crea_custom_dataset('/home/silvia/Desktop/data_D_class47_54.hdf5', '/home/silvia/Desktop/metadata_D_class47_54.csv')
"""