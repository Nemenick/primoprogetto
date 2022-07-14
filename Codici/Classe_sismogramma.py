import dask.dataframe as dd
import numpy as np
import h5py
import matplotlib.pyplot as plt


class Classe_Dataset:
    def letturacsv(self, percorsocsv, coltot):  # coltot = ["trace_name","trace polarity", ...]
        self.percorsocsv = percorsocsv
        datd = dd.read_csv(self.percorsocsv, usecols=coltot)
        self.allmetadata = {}
        for i in coltot:
            self.allmetadata[i] = np.array(datd[i])
        # mi faccio il dizionario metadata["tracename"][1] etc
        # print(self.metadata["trace_name"])

    def selezioneDati(self, percorsohdf5):  # colselezione faccio selezione solo dei dati che servono
        filehdf5 = h5py.File(percorsohdf5, 'r')
        dataset = filehdf5.get("data")
        print("\ndatasetORI", dataset)
        # nomidata = list(dataset.keys())                                  # Mi sono salvato i nomi di tutti i dataset
        nomidata = self.allmetadata["trace_name"]
        print(nomidata[2])
        sismogramma = []
        self.metadata = {}
        for key in self.allmetadata:            # creo dataset selezionato ma che ha gli stessi metadata del completo
            self.metadata[key] = []
        for i in range(len(nomidata)):
            if (self.allmetadata["trace_polarity"][i] != 'undecidable'):        # TODO condizione da aggiornare
                sismogramma.append(dataset.get(nomidata[i]))
            for key in self.metadata:
                self.metadata[key].append(self.allmetadata[key][i])
                # P_times.append(trace_P_arrival_sample[i])  # Solo quelli con polarity definita
                # P_polarity.append(trace_polarity[i])
                sismogramma[-1] = sismogramma[-1][2]  # TODO incredibilmente ci mette di meno!
        sismogramma = np.array(sismogramma)

csv = 'C:/Users/HP_i3-7200U/Desktop/Simple_dataset/metadata/metadata_Instance_events_10k.csv'
coltot = ["trace_name", "station_channels", "trace_P_arrival_sample", "trace_polarity"] # trace_name,station_channels needed
Dataset_1 = Classe_Dataset()
Dataset_1.letturacsv(csv, coltot)