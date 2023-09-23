from Classe_sismogramma_v3 import ClasseDataset

hdf5in_ori = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s.hdf5'
csvin_ori = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s.csv'

dat = ClasseDataset()
dat.leggi_custom_dataset(hdf5in_ori, csvin_ori)
print(type(dat.metadata["trace_name"]))

nomi = ["q","j","s"]
nomi = nomi + list(dat.metadata["trace_name"])
print(nomi[0:100])
