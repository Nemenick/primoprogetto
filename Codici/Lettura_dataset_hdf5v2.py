import h5py
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd

visualizza = 4

# TOdo lettura file CSV
datd = dd.read_csv('C:/Users/HP_i3-7200U/Desktop/Simple_dataset/metadata/metadata_Instance_events_10k.csv', usecols=["trace_name", "trace_P_arrival_sample", "trace_polarity"])
# print(np.array(datd["source_id"]))
trace_name = np.array(datd["trace_name"])
trace_P_arrival_sample = np.array(datd["trace_P_arrival_sample"])                   # Gli originali
trace_polarity = np.array(datd["trace_polarity"])
P_times = []
P_polarity = []
input("ciao")
# for i in range(len(trace_P_arrival_sample)):
#     if trace_polarity[i] != 'undecidable':
#         P_times.append(trace_P_arrival_sample[i])                                  # Solo quelli con polarity definita
#         P_polarity.append(trace_polarity[i])					     # Lo faccio Dopo

# Todo Lettura file HDF5
filehdf5 = h5py.File('C:/Users/HP_i3-7200U/Desktop/Simple_dataset/data/Instance_events_gm_10k.hdf5', 'r')
print("primofile", list(filehdf5.items()), filehdf5.keys())          # restituisce le key (per accedere agli elementi)
# formato = filehdf5.get('data_format')
# print(formato)

dataset = filehdf5.get("data")
print("\ndatasetORI", dataset)
# nomidata = list(dataset.keys())                                        # Mi sono salvato i nomi di tutti i dataset
nomidata = trace_name
print(nomidata[2])
sismogramma = []
for i in range(len(nomidata)):
    if trace_polarity[i] != 'undecidable':
        sismogramma.append(dataset.get(nomidata[i]))
        P_times.append(trace_P_arrival_sample[i])                           # Solo quelli con polarity definita
        P_polarity.append(trace_polarity[i])
        sismogramma[-1] = sismogramma[-1][2]                                # TODO incredibilmente ci mette di meno!
sismogramma = np.array(sismogramma)

print("lunghezze", len(sismogramma), len(P_times), len(sismogramma))
print("sismogramma.shape", sismogramma.shape)

sismogramma_centrato = []
for i in range(len(sismogramma)):
    sismogramma_centrato.append(sismogramma[i][P_times[i]-100:P_times[i]+100])
sismogramma_centrato = np.array(sismogramma_centrato)
print("shape_centrato", sismogramma_centrato.shape)
for i in range(visualizza):
    plt.plot(range(200), sismogramma_centrato[i*10+3])
    plt.axvline(x=100, c="r", ls="--")
    plt.title(str(P_times[i*10+3]) + " " + P_polarity[i*10+3])
    # plt.savefig("Metodo1_centrato_"+str(i*10+3))
    # plt.clf()
    plt.show()

# TODO mi faccio plottare i primi 50 che hanno polarità known senza selezionare mentre faccio input
# sismogramma = []
# for i in range(len(nomidata)):
#     sismogramma.append(dataset.get(nomidata[i]))            # caricp TUTTE le forme d'onda
# plottati = 0
# i = 0
# while plottati < visualizza:  # Plot dei primi 50 con polarità definita
#     if trace_polarity[i] != "undecidable":
#         plt.plot(range(12000), sismogramma[i][2])
#         plt.axvline(x=trace_P_arrival_sample[i], c="r", ls="--")
#         plt.title(str(trace_P_arrival_sample[i]) + " " + trace_polarity[i])
#         plt.savefig("Metodo2_"+str(plottati) + "_" + str(i))
#         print("Metodo2_"+str(plottati) + "_" + str(i))
#         plottati = plottati + 1
#         plt.clf()
#     i = i + 1

# TODO accedere dati su hdf5 tramite nomi presenti nel file CSV (tracename)

filehdf5.close()
