import pandas as pd
from Classe_sismogramma_v3 import ClasseDataset
import numpy as np

hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s.hdf5'
csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s.csv'

hdf5out = '/home/silvia/Desktop/SOM_errori/Tentativo_27/data_Velocimeter_Buone_4s_test_sbagliati27.hdf5'  # TODO
csvout = '/home/silvia/Desktop/SOM_errori/Tentativo_27/metadata_Velocimeter_Buone_4s_test_sbagliati27.csv'

txt_data = '/home/silvia/Desktop/SOM_errori/Tentativo_27/data_Velocimeter_Buone_4s_test_sbagliati27.txt'

Datain = ClasseDataset()
Datain.leggi_custom_dataset(hdf5in, csvin)
print("Len di tutto", Datain.sismogramma.shape, len(Datain.metadata["trace_name"]))

estremi_test = [43, 45, 9.5, 11.8]
indici_test = []
for k in range(len(Datain.sismogramma)):
    if estremi_test[0] < Datain.metadata['source_latitude_deg'][k] < estremi_test[1] and estremi_test[2] \
            < Datain.metadata['source_longitude_deg'][k] < estremi_test[3]:
        indici_test.append(k)

data = []
dizio_metadata = {}
for key in Datain.metadata.keys():
    dizio_metadata[key] = []

for i in indici_test:
    data.append(Datain.sismogramma[i])
    # print(Datain.sismogramma[i])
    for key in Datain.metadata.keys():
        dizio_metadata[key].append(Datain.metadata[key][i])
print(i, Datain.sismogramma[i],type(i))
print(len(i))
data_return = ClasseDataset()
data_return.sismogramma = np.array(data)
data_return.metadata = dizio_metadata
data_return.centrato = Datain.centrato
data_return.demeaned = Datain.demeaned

# Datatest = Datain.seleziona_indici(indici_test)
print("Len di test", data_return.sismogramma.shape, len(data_return.metadata["trace_name"]))
print("len data list", len(data), len(data[1]))
"""
path = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi'
tentativi = "27"
predizioni = pd.read_csv(path + '/' + tentativi + '/Predizioni_test_tentativo_' + tentativi + '.csv')

for i in range(len(predizioni["delta_test"])):
    if predizioni["delta_test"][i] >= 0.5:
        vettore_indici.append(i)

Dataout = Datatest.seleziona_indici(vettore_indici)
print(Dataout.metadata["centrato"], "\n##########", Dataout.centrato)
Dataout.crea_custom_dataset(hdf5out, csvout)
Dataout.to_txt(txt_data)"""
