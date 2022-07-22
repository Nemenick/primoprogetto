import dask.dataframe as dd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings




class Classe_Dataset:
    """def letturacsv(self, percorsocsv, coltot):  # coltot = ["trace_name","trace polarity", ...]
        self.percorsocsv = percorsocsv

        datd = dd.read_csv(self.percorsocsv, usecols=coltot)
        self.allmetadata = {}
        for i in coltot:                # genera metadata["colname"] = np.array["colname"]
            self.allmetadata[i] = np.array(datd[i])
        for key in self.allmetadata:
            print(key, self.allmetadata[key])
        # creo il dizionario metadata["tracename"][1] etc
        # print(self.metadata["trace_name"])"""

    def __init__(self):
        self.centrato = False           # dice se ho tagliato e centrato la finestra temporale

    def acquisisci_new(self, percorsohdf5, percorsocsv, coltot, nomi_selezionati, **kwargs):
        """
        Acquisisce e seleziona tracce del file hdf5 e csv
        e salva in file csv nomi_selezionati e indici delle tracce selezionate
        ATTENTO! (non creo un custom dataset di trace, ma solo salvo in csv lista di quelle da leggere)
        kwargs per chiamare Finestra    # TODO implementa
        """
        self.percorsocsv = percorsocsv
        start = time.perf_counter()
        # FIXME engine"python" (lentissimo): dava problemi la riga 33114, l'ho skippata e legge ma dice che e too large-
        # datd = dd.read_csv(self.percorsocsv, usecols=coltot, engine="python", on_bad_lines="skip", sample=10**8, assume_missing=True)
        datd = dd.read_csv(self.percorsocsv, usecols=coltot)
        print(time.perf_counter() - start)
        self.allmetadata = {}
        for i in coltot:  # genera metadata["colname"] = np.array["colname"]
            self.allmetadata[i] = np.array(datd[i])                                    # LEGGO CSV
            print(i, time.perf_counter() - start)
        for key in self.allmetadata:
            print(key, self.allmetadata[key])
        # creo il dizionario metadata["tracename"][1] etc
        # print(self.metadata["trace_name"])
        print(time.perf_counter() - start)

        filehdf5 = h5py.File(percorsohdf5, 'r')
        dataset = filehdf5.get("data")
        # print("\ndatasetORI", dataset)
        # nomidata = list(dataset.keys())                             # Mi sono salvato i nomi di tutti i dataset
        nomidata = self.allmetadata["trace_name"]                     # Presi dal file CSV
        # print(nomidata)
        self.sismogramma = []
        self.metadata = {}
        self.indice_csv = []
        for key in self.allmetadata:            # creo dataset selezionato ma che ha gli stessi metadata del completo
            self.metadata[key] = []
        for i in range(len(nomidata)):
            if self.allmetadata["trace_polarity"][i] != 'undecidable' \
                    and (self.allmetadata["station_channels"][i] == "HH" or
                         self.allmetadata["station_channels"][i] == "EH"):        # TODO condizione da aggiornare
                self.sismogramma.append(dataset.get(nomidata[i]))
                self.sismogramma[-1] = self.sismogramma[-1][2]  # Componente z, ci mette di meno
                self.indice_csv.append(i)
                for key in self.metadata:
                    self.metadata[key].append(self.allmetadata[key][i])
        print("\nshape_prima di ridimensionare\n",len(self.sismogramma),len(self.sismogramma[0]))
        self.sismogramma = np.array(self.sismogramma)
        self.indice_csv = np.array(self.indice_csv)
        pd_names = pd.DataFrame({"trace_name": self.metadata["trace_name"],
                                 "indice_csv": self.indice_csv})  # basta indice_csv
        pd_names.to_csv(nomi_selezionati, index=False)
        # pd_names = tracenames':[], 'indice file csv'
        print("shape", self.sismogramma.shape, len(self.metadata["trace_P_arrival_sample"]))

    def acquisisci_old(self, percorsohdf5, percorsocsv, coltot, percorso_nomi):
        """"
        Acquisisce le tracce presenti in file hdf5 e csv che sono nominate nel file percorso nomi
        che già sono stati selezionati in precedenza con acquisici_new
        """

        self.percorsocsv = percorsocsv
        datd = dd.read_csv(self.percorsocsv, usecols=coltot)
        self.allmetadata = {}
        for i in coltot:  # genera metadata["colname"] = np.array["colname"]
            self.allmetadata[i] = np.array(datd[i])                                 # LEGGO CSV
        for key in self.allmetadata:
            print(key, self.allmetadata[key])
        # creo il dizionario metadata["tracename"][1] etc
        # print(self.metadata["trace_name"])

        filehdf5 = h5py.File(percorsohdf5, 'r')
        dataset = filehdf5.get("data")
        print("\ndatasetORI", dataset)
        # nomidata = list(dataset.keys())                                  # Mi sono salvato i nomi di tutti i dataset
        datnomi = dd.read_csv(percorso_nomi, usecols=["trace_name", "indice_csv"])
        nomidata = np.array(datnomi["trace_name"])
        self.indice_csv = np.array(datnomi["indice_csv"])
        print(nomidata)
        self.sismogramma = []
        self.metadata = {}

        for key in self.allmetadata:            # creo dataset selezionato ma che ha gli stessi metadata del completo
            self.metadata[key] = []
        for i in range(len(nomidata)):
            self.sismogramma.append(dataset.get(nomidata[i]))
            self.sismogramma[-1] = self.sismogramma[-1][2]  # Componente z, ci mette di meno
            for key in self.metadata:
                self.metadata[key].append(self.allmetadata[key][self.indice_csv[i]])
        self.sismogramma = np.array(self.sismogramma)

        print("shape", self.sismogramma.shape, len(self.metadata["trace_P_arrival_sample"]))

    def crea_custom_dataset(self, percorsohdf5in, percorsocsvin, percorsohdf5out, percorsocsvout_pandas, coltot, **kwargs):
        """
        creo il dataset che mi piace, selezionando alcune tracce di hdf5,csv in e mettendole in out
        kwargs per chiamare Finestra # TODO implementa
        """
        start = time.perf_counter()
        self.percorsocsv = percorsocsvin
        datd = dd.read_csv(self.percorsocsv, usecols=coltot, engine="python", on_bad_lines="skip",
                           sample=10**8, assume_missing=True)
        self.allmetadata = {}
        for i in coltot:  # genera metadata["colname"] = np.array["colname"]
            self.allmetadata[i] = np.array(datd[i])  # LEGGO CSV
            print(i, time.perf_counter() - start)
        for key in self.allmetadata:
            print(key, self.allmetadata[key])
        # creo il dizionario metadata["tracename"][1] etc
        # print(self.metadata["trace_name"])

        filehdf5 = h5py.File(percorsohdf5in, 'r')
        dataset = filehdf5.get("data")
        print("\ndatasetORI", dataset)
        # nomidata = list(dataset.keys())                             # Mi sono salvato i nomi di tutti i dataset
        nomidata = self.allmetadata["trace_name"]  # Presi dal file CSV
        print(nomidata)
        self.sismogramma = []
        self.metadata = {}
        for key in self.allmetadata:  # creo dataset selezionato ma che ha gli stessi metadata del completo
            self.metadata[key] = []
        for i in range(len(nomidata)):
            if self.allmetadata["trace_polarity"][i] != 'undecidable' \
                    and (self.allmetadata["station_channels"][i] == "HH" or
                         self.allmetadata["station_channels"][i] == "EH"):  # TODO condizione da aggiornare
                if i % 5000 == 0:
                    print("sto caricando il sismogramma ", i)
                self.sismogramma.append(dataset.get(nomidata[i]))
                self.sismogramma[-1] = self.sismogramma[-1][2]  # Componente z, ci mette di meno
                for key in self.metadata:
                    self.metadata[key].append(self.allmetadata[key][i])
        self.sismogramma = np.array(self.sismogramma)
        print("shape", self.sismogramma.shape, len(self.metadata["trace_P_arrival_sample"]))
        filehdf5.close()
        print("\n\nFINE CARICAMENTODATI", time.perf_counter() - start)

        startp = time.perf_counter()
        filehdf5 = h5py.File(percorsohdf5out, 'w')
        filehdf5.create_dataset(name='dataset1', data=self.sismogramma)
        print("ho creato hdf5")
        datapandas = pd.DataFrame.from_dict(self.metadata)
        datapandas.to_csv(percorsocsvout_pandas, index=False)
        filehdf5.close()
        print("\n\n PANDAS HA AGITO", time.perf_counter() - startp)

    def leggi_custom_dataset(self, percorsohdf5, percorsocsv):
        """
        legge TUTTE le tracce di questo dataset
        le ho salvate(solo componenteZ) in un unico dataset nel file percorsohdf5
        """

        start = time.perf_counter()

        filehdf5 = h5py.File(percorsohdf5, 'r')
        self.sismogramma = filehdf5.get("dataset1")
        self.sismogramma = np.array(self.sismogramma)
        print("ho caricato hdf5", time.perf_counter()-start)
        datd = dd.read_csv(percorsocsv, dtype={"trace_P_arrival_sample": int})
        # non metto engine, assume missinng etc perchè questi selezionati sembrano buoni
        print("ho letto csv", time.perf_counter()-start)
        self.metadata = {}
        for key in datd:
            self.metadata[key] = np.array(datd[key])
            print("ho caricato la key ", key, time.perf_counter() - start)
        print(self.sismogramma.shape, len(self.sismogramma))

    def Finestra(self, semiampiezza=0):
        """
            taglia e se necessario centra la finestra
            semiampiezza: numero di samples (0.01s) es 100 per finestra di 2 sec
        """
        sismogramma = [0 for i in range(len(self.sismogramma))]

        if self.centrato:
            if len(self.sismogramma[0]) > 2 * semiampiezza:
                centro = len(self.sismogramma[0]) // 2
                for i in range(len(self.sismogramma)):
                    sismogramma[i] = self.sismogramma[i][centro - semiampiezza:
                                                         centro + semiampiezza]
                self.sismogramma = np.array(sismogramma)
            else:
                print("\nE' gia centrato e gia con finsetra + piccola della richiesta")
                print("Non ho fatto niente\n")

        else:
            for i in range(len(self.sismogramma)):
                if self.metadata["trace_P_arrival_sample"][i] > semiampiezza:
                    sismogramma[i] = self.sismogramma[i][self.metadata["trace_P_arrival_sample"][i] - semiampiezza:
                                                         self.metadata["trace_P_arrival_sample"][i] + semiampiezza]
                else:
                    stringa = "#"
                    for _ in range(300):
                        stringa = stringa + "#"
                    warnings.warn("\n"+stringa+"\nATTENTO, SCEGLI FINESTRA PIU PICCOLA!,"
                                               "continuo senza centrare nulla\n"+stringa)
                    print("semiampiezza = ", semiampiezza, "ArrivoP = ", self.metadata["trace_P_arrival_sample"][i])
                    return 1

            self.sismogramma = np.array(sismogramma)

            self.centrato = True

    def to_txt(self, percorsohdf5, percorsocsv, coltot, nomi_selezionati, txt_data, txt_metadata):
        self.acquisisci_new(percorsohdf5, percorsocsv, coltot=coltot, nomi_selezionati=nomi_selezionati)
        # print("\n\nVA BENE?", self.sismogramma)
        np.savetxt(txt_data, self.sismogramma, fmt='%.5e')
        metadata_txt = pd.DataFrame.from_dict(self.metadata)
        metadata_txt.to_csv(txt_metadata, index=False, sep='\t')
        # df.to_csv(r'c:\data\pandas.txt', header=None, index=None, sep='\t', mode='a')

    def plotta(self, visualizza, semiampiezza=None, namepng=None):
        if len(self.sismogramma) < visualizza:
            print("lunghezza sismogramma < sismogrammi da visualizzare")
            return 1

        if self.centrato:
            if semiampiezza is None or semiampiezza > len(self.sismogramma[0])//2:
                semiampiezza = len(self.sismogramma[0])//2
            for i in range(visualizza):
                lung = len(self.sismogramma[0])
                plt.plot(range(2*semiampiezza), self.sismogramma[i][lung//2 - semiampiezza:
                                                                    lung//2 + semiampiezza])
                plt.axvline(x=semiampiezza, c="r", ls="--")
                stringa = ""
                for key in self.metadata:
                    stringa = stringa + str(self.metadata[key][i]) + " "
                # stringa = stringa + str(self.indice_csv[i])
                plt.title(stringa)
                plt.savefig(namepng + "_" + str(i))  # TODO if name = None non fare niente
                plt.clf()
                # plt.show()
        else:
            semiampiezza_ori = semiampiezza
            for i in range(visualizza):
                if semiampiezza_ori is None or semiampiezza_ori > self.metadata["trace_P_arrival_sample"][i]:
                    semiampiezza = self.metadata["trace_P_arrival_sample"][i]-1
                else:
                    semiampiezza = semiampiezza_ori
                # print(self.metadata["trace_P_arrival_sample"][i])
                plt.plot(range(2*semiampiezza),
                         self.sismogramma[i][self.metadata["trace_P_arrival_sample"][i] - semiampiezza:
                                             self.metadata["trace_P_arrival_sample"][i] + semiampiezza])
                plt.axvline(x=semiampiezza, c="r", ls="--")
                stringa = ""
                for key in self.metadata:
                    stringa = stringa + str(self.metadata[key][i]) + " "
                # stringa = stringa + str(self.indice_csv[i])
                plt.title(stringa)
                plt.savefig(namepng + "_" + str(i))       # TODO if name = None non fare niente
                plt.clf()
                # plt.show()



csvin = 'C:/Users/GioCar/Desktop/Simple_dataset/metadata/metadata_Instance_events_10k.csv'
hdf5in = 'C:/Users/GioCar/Desktop/Simple_dataset/data/Instance_events_counts_10k.hdf5'
csvout = 'C:/Users/GioCar/Desktop/Simple_dataset/metadata_Instance_events_selected_Polarity_Velocimeter.csv'
hdf5out = 'C:/Users/GioCar/Desktop/Simple_dataset/data_selected_Polarity_Velocimeter.hdf5'
txt_data = "C:/Users/GioCar/Desktop/txt_tracce.txt"
txt_metadata = "C:/Users/GioCar/Desktop/txt_metadata.txt"
coltot = ["trace_name", "station_channels", "trace_P_arrival_sample", "trace_polarity",
          "trace_P_uncertainty_s", "source_magnitude", "source_magnitude_type"]
nomi = "Selezionati.csv"


Dataset_1 = Classe_Dataset()
print(Dataset_1.centrato)
Dataset_1.acquisisci_old(percorsohdf5=hdf5in, percorsocsv=csvin, coltot=coltot, percorso_nomi=nomi)
Dataset_1.Finestra(1000000)
Dataset_1.plotta(50,semiampiezza=100,namepng="prova")
if __name__ == "main":
    print("ci")
    # csvin = 'C:/Users/GioCar/Desktop/Simple_dataset/metadata/metadata_Instance_events_10k.csv'
    # hdf5in = 'C:/Users/GioCar/Desktop/Simple_dataset/data/Instance_events_counts_10k.hdf5'
    # csvout = 'C:/Users/GioCar/Desktop/Simple_dataset/metadata_Instance_events_selected_Polarity_Velocimeter.csv'
    # hdf5out = 'C:/Users/GioCar/Desktop/Simple_dataset/data_selected_Polarity_Velocimeter.hdf5'
    # txt_data = "C:/Users/GioCar/Desktop/txt_tracce.txt"
    # txt_metadata = "C:/Users/GioCar/Desktop/txt_metadata.txt"
    # coltot = ["trace_name", "station_channels", "trace_P_arrival_sample", "trace_polarity",
    #           "trace_P_uncertainty_s", "source_magnitude", "source_magnitude_type"]
    # nomi = "Selezionati.csv"
    #
    # trace_name,station_channels needed
    # Dataset_1 = Classe_Dataset()
    # Dataset_1.to_txt(hdf5in, csvin, ["trace_name", "station_channels", "trace_P_arrival_sample",
    #                                  "trace_polarity", "source_magnitude"], txt_data, txt_metadata)
    # Dataset_1.leggi_custom_dataset(hdf5out,csvout)
    # Dataset_1.crea_custom_dataset(hdf5in,csvin,hdf5out,csvout,coltot=coltot)
    # Dataset_1.acquisisci_new(percorsohdf5=hdf5in, percorsocsv=csvin, coltot=coltot, nomi_selezionati=nomi)
    # Dataset_1.plotta(visualizza=30, namepng="Dataset_counts")
    # Dataset_1.acquisisci_old(percorsohdf5=hdf5, percorsocsv=csv, coltot=coltot, percorso_nomi=nomi)
    # Dataset_1.plotta(visualizza=5, namepng="/home/silvia/Desktop/Figure_Large_Custom_dataset/Custom_Large_dataset")
