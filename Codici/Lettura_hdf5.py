import h5py
import numpy as np
import matplotlib.pyplot as plt

m1 = np.random.random(size=(12, 5, 1))
print(m1)
print(m1.shape)
print(type(m1))

primofile = h5py.File('C:/Users/HP_i3-7200U/Desktop/filemio.hdf5', 'w')             # creo file hdf5
primofile.create_dataset(name='dataset1', data=m1)                            # cre

G1 = primofile.create_group(name='group1')                                       # creo gruppo
G1.create_dataset(name='data1_1', data=m1+1)
print('primofile', type(primofile))
print('G1', type(G1))

G11 = G1.create_group(name='group1_1')
G11.create_dataset(name='dataset1_1_1', data=m1+11)

primofile.close()
print('G11', type(G11))

primofile = h5py.File('C:/Users/HP_i3-7200U/Desktop/filemio.hdf5', 'r')
print(primofile.keys())
key1 = primofile.get("group1")
print(key1)
data1 = primofile.get("dataset1")
data1 = np.array(data1)
print(data1.shape)
print(type(data1))


lista = list(primofile.items())                        # IMPORTANTISSIMO fa vedere tutti i dataset o gruppi appartenenti
print(lista)


primofile.close()


primofile = h5py.File('C:/Users/HP_i3-7200U/Desktop/Simple_dataset/data/Instance_events_counts_10k.hdf5', 'r')
print("KEYS", primofile.keys())                                                                     # restituisce le key
formato = primofile.get('data_format')
print(formato)

lista = list(primofile.items())
print(lista)
dataset = primofile.get("/data")
print("datasetORI", dataset)
nomi_data = np.array(dataset)
print(nomi_data[0:9])                                                   # todo Mi sono salvato i nomi di tutti i dataset
print("datasetORI", dataset)
sismogramma = 0
#for i in nomi_data[0:9]:
sismogramma = np.array(dataset.get(nomi_data[24]))
print("sismogramma", sismogramma)
plt.plot(range(12000), sismogramma[2])
plt.show()
# lista = list(dataset.items())
# print("lista dataset", lista)
# dataset = np.array(dataset)
# print(dataset.shape)
print(type(dataset))
primofile.close()
