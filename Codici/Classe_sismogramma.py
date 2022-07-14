import dask.dataframe as dd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

# TODO invece di fare dizionario fai pd.dataframe


class Classe_Dataset:
    def letturacsv(self, percorsocsv, coltot):  # coltot = ["trace_name","trace polarity", ...]
        self.percorsocsv = percorsocsv
        datd = dd.read_csv(self.percorsocsv, usecols=coltot)
        self.allmetadata = {}
        for i in coltot:                # genera metadata["colname"] = np.array["colname"]
            self.allmetadata[i] = np.array(datd[i])
        for key in self.allmetadata:
            print(key, self.allmetadata[key])
        # creo il dizionario metadata["tracename"][1] etc
        # print(self.metadata["trace_name"])

    # TODO fai che si chiama lettura scv autonomamente
    def acquisisci_new(self, percorsohdf5):  # colselezione faccio selezione solo dei dati che servono
        filehdf5 = h5py.File(percorsohdf5, 'r')
        dataset = filehdf5.get("data")
        print("\ndatasetORI", dataset)
        # nomidata = list(dataset.keys())                                  # Mi sono salvato i nomi di tutti i dataset
        nomidata = self.allmetadata["trace_name"]
        print(nomidata)
        self.sismogramma = []
        self.metadata = {}

        for key in self.allmetadata:            # creo dataset selezionato ma che ha gli stessi metadata del completo
            self.metadata[key] = []
        for i in range(len(nomidata)):
            if self.allmetadata["trace_polarity"][i] != 'undecidable' \
                    and (self.allmetadata["station_channels"][i] == "HH" or
                         self.allmetadata["station_channels"][i] == "EH"):        # TODO condizione da aggiornare
                self.sismogramma.append(dataset.get(nomidata[i]))
                self.sismogramma[-1] = self.sismogramma[-1][2]  # Componente z, ci mette di meno
                for key in self.metadata:
                    self.metadata[key].append(self.allmetadata[key][i])
                    # P_times.append(trace_P_arrival_sample[i])  # Solo quelli con polarity definita
                    # P_polarity.append(trace_polarity[i])
        self.sismogramma = np.array(self.sismogramma)
        pd_names = pd.DataFrame({"trace_name": self.metadata["trace_name"]})
        pd_names.to_csv("Selezionati.csv", index=False)
        # TODO mi devo far dire a che posizione è nel file csv, per riprenderlo da lì
        # pd_names = tracenames':[], 'indice file csv'
        print("shape", self.sismogramma.shape, len(self.metadata["trace_P_arrival_sample"]))

    # TODO fai che si chiama lettura scv autonomamente (passa percorso csv)
    # legge da questo cvs e hdf5 le tracce presenti in percorso nomi
    def acquisisci_old(self, percorsohdf5, percorso_nomi):  # colselezione faccio selezione solo dei dati che servono
        filehdf5 = h5py.File(percorsohdf5, 'r')
        dataset = filehdf5.get("data")
        print("\ndatasetORI", dataset)
        # nomidata = list(dataset.keys())                                  # Mi sono salvato i nomi di tutti i dataset
        datd = dd.read_csv(percorso_nomi, usecols=["trace_name", "numero_csv"])
        nomidata = np.array(datd["trace_name"])  # FIXME RIVEDI SE VA
        indice_csv = np.array(datd["numero_csv"])
        print(nomidata)
        self.sismogramma = []
        self.metadata = {}
    # TODO fai CHE SI CHIAMA DA SOLO letturaCSV(scongiuro di non sapere le key da usare)
        for key in self.allmetadata:            # creo dataset selezionato ma che ha gli stessi metadata del completo
            self.metadata[key] = []
        for i in range(len(nomidata)):
            self.sismogramma.append(dataset.get(nomidata[i]))
            self.sismogramma[-1] = self.sismogramma[-1][2]  # Componente z, ci mette di meno
            for key in self.metadata:
                self.metadata[key].append(self.allmetadata[key][indice_csv[i]])  # FIXME va bene indicecsv??
        self.sismogramma = np.array(self.sismogramma)

        print("shape", self.sismogramma.shape, len(self.metadata["trace_P_arrival_sample"]))

    def plotta(self, visualizza):
        if len(self.sismogramma < visualizza):
            print("lunghezza sismogramma < visualizza")
            return 1
        for i in range(visualizza):                 # TODO aggiungi qui sotto un # per far printare traccia tutta
            plt.plot(range(200), self.sismogramma[i][self.metadata["trace_P_arrival_sample"][i] - 100:self.metadata["trace_P_arrival_sample"][i] + 100])
            plt.axvline(x=100, c="r", ls="--")
            stringa = ""
            for key in self.metadata:
                stringa = stringa + str(self.metadata[key][i]) + " "
            plt.title(stringa)
            plt.savefig("Metodo1_centrato_"+str(i))
            plt.clf()
            # plt.show()


csv = 'C:/Users/HP_i3-7200U/Desktop/Simple_dataset/metadata/metadata_Instance_events_10k.csv'
hdf5 = 'C:/Users/HP_i3-7200U/Desktop/Simple_dataset/data/Instance_events_counts_10k.hdf5'
coltot = ["trace_name", "station_channels", "trace_P_arrival_sample", "trace_polarity",
          "trace_P_uncertainty_s", "source_magnitude", "source_magnitude_type"]
# trace_name,station_channels needed
Dataset_1 = Classe_Dataset()
Dataset_1.letturacsv(csv, coltot)
Dataset_1.acquisisci_new(hdf5)  # TODO fare acquisisci_old, non selezione quelli da acquisire ma tracenames da file xls
# Dataset_1.plotta(visualizza=0)
