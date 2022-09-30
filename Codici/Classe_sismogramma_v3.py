import dask.dataframe as dd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
import os
# import openpyxl


class ClasseDataset:

    def __init__(self):
        """
        inserisci qui tutte le proprietà di classe con i ripsettivi tipi
        """
        self.centrato = False           # dice se ho tagliato e centrato la finestra temporale
        self.demeaned = False           # dice se la media è tolta. Due tipi medie : sarà stringa, "rumore" o "totale"

        self.sismogramma = np.array([])     # np.array (,)
        self.metadata = {}                  # dizionario di lista, non np.array (non so come li tratta pandas)
        self.classi = []                    # lista di int

    def acquisisci_new(self, percorsohdf5, percorsocsv, col_tot, nomi_selezionati):
        """
        Acquisisce e seleziona tracce del file hdf5 e csv
        e salva in file csv i nomi_selezionati e indici delle tracce selezionate
        ATTENTO! (non creo un custom dataset di trace, ma solo salvo in csv lista di quelle da leggere)
        """
        percorsocsv = percorsocsv
        start = time.perf_counter()
        # FIXME engine"python" (lentissimo): dava problemi la riga 33114, l'ho skippata e legge ma dice che e too large-
        # datd = dd.read_csv(self.percorsocsv, usecols=coltot, engine="python", on_bad_lines="skip",
        #                    sample=10 ** 8, assume_missing=True)
        datd = dd.read_csv(percorsocsv, usecols=col_tot)

        allmetadata = {}
        for i in col_tot:  # genera metadata["colname"] = np.array["colname"]
            allmetadata[i] = np.array(datd[i])                                    # LEGGO CSV
            print(i, time.perf_counter() - start)
        for key in allmetadata:
            print(key, allmetadata[key])
        # creo il dizionario metadata["tracename"][1] etc
        # print(self.metadata["trace_name"])
        print(time.perf_counter() - start)

        filehdf5 = h5py.File(percorsohdf5, 'r')
        dataset = filehdf5.get("data")
        # print("\ndatasetORI", dataset)
        # nomidata = list(dataset.keys())                             # Mi sono salvato i nomi di tutti i dataset
        nomidata = allmetadata["trace_name"]                     # Presi dal file CSV
        # print(nomidata)
        self.sismogramma = []
        self.metadata = {}
        indice_csv = []
        for key in allmetadata:            # creo dataset selezionato ma che ha gli stessi metadata del completo
            self.metadata[key] = []
        for i in range(len(nomidata)):
            if i % 10000 == 0:
                print("sto analizzando il sismogramma ", i)
            if allmetadata["trace_polarity"][i] == 'positive' \
                    and (allmetadata["station_channels"][i] == "HH" or
                         allmetadata["station_channels"][i] == "EH"):        # TODO condizione da aggiornare
                self.sismogramma.append(dataset.get(nomidata[i]))
                self.sismogramma[-1] = self.sismogramma[-1][2]  # Componente z, ci mette di meno
                indice_csv.append(i)
                for key in self.metadata:
                    self.metadata[key].append(allmetadata[key][i])
        print("\nshape_prima di ridimensionare\n", len(self.sismogramma), len(self.sismogramma[0]))
        self.sismogramma = np.array(self.sismogramma)
        print("shape dopo", self.sismogramma.shape, len(self.metadata["trace_P_arrival_sample"]))
        filehdf5.close()
        indice_csv = np.array(indice_csv)
        pd_names = pd.DataFrame({"trace_name": self.metadata["trace_name"],
                                 "indice_csv": indice_csv})  # basta indice_csv
        pd_names.to_csv(nomi_selezionati, index=False)
        # pd_names = tracenames':[], 'indice file csv'
        print("HO creato il file nomi_selezionati")

    def acquisisci_old(self, percorsohdf5, percorsocsv, col_tot, percorso_nomi):
        """"
        Acquisisce le tracce presenti in file hdf5 e csv che sono nominate nel file percorso nomi
        che già sono stati selezionati in precedenza con acquisici_new
        secondo la "sintassi" dettata da INSTANCE, non sono custom_dataset
        """

        datd = dd.read_csv(percorsocsv, usecols=col_tot)
        allmetadata = {}
        for i in col_tot:  # genera metadata["colname"] = np.array["colname"]
            allmetadata[i] = np.array(datd[i])                                 # LEGGO CSV
        for key in allmetadata:
            print(key, allmetadata[key])
        # creo il dizionario metadata["tracename"][1] etc
        # print(self.metadata["trace_name"])

        filehdf5 = h5py.File(percorsohdf5, 'r')
        dataset = filehdf5.get("data")
        # print("\ndatasetORI", dataset)
        # nomidata = list(dataset.keys())                                  # Mi sono salvato i nomi di tutti i dataset
        datnomi = dd.read_csv(percorso_nomi, usecols=["trace_name", "indice_csv"])
        nomidata = np.array(datnomi["trace_name"])
        indice_csv = np.array(datnomi["indice_csv"])
        print(nomidata)
        self.sismogramma = []
        self.metadata = {}

        for key in allmetadata:            # creo dataset selezionato ma che ha gli stessi metadata del completo
            self.metadata[key] = []
        for i in range(len(nomidata)):
            self.sismogramma.append(dataset.get(nomidata[i]))
            self.sismogramma[-1] = self.sismogramma[-1][2]  # Componente z, ci mette di meno
            for key in self.metadata:
                self.metadata[key].append(allmetadata[key][indice_csv[i]])
        self.sismogramma = np.array(self.sismogramma)

        print("shape", self.sismogramma.shape, len(self.metadata["trace_P_arrival_sample"]))

    def crea_custom_dataset(self, percorsohdf5out, percorsocsvout_pandas):
        """
        creo il dataset che mi piace, selezionando alcune tracce di hdf5,csv in e mettendole in out
        serve per non caricare ogni volta tutte le tracce
        """

        startp = time.perf_counter()
        filehdf5 = h5py.File(percorsohdf5out, 'w')
        print("sto creando hdf5")
        filehdf5.create_dataset(name='dataset1', data=self.sismogramma)
        print("ho creato hdf5")
        dizio = self.metadata
        dizio["centrato"] = self.centrato
        dizio["demeaned"] = self.demeaned
        datapandas = pd.DataFrame.from_dict(dizio)
        datapandas.to_csv(percorsocsvout_pandas, index=False)
        filehdf5.close()
        print("\n\n PANDAS HA AGITO", time.perf_counter() - startp)

    def leggi_custom_dataset(self, percorsohdf5, percorsocsv):
        """
        legge TUTTE le tracce di questo custom_dataset
        le ho salvate(solo componenteZ) in un unico dataset nel file percorsohdf5
        """
        start = time.perf_counter()
        filehdf5 = h5py.File(percorsohdf5, 'r')
        self.sismogramma = filehdf5.get("dataset1")
        self.sismogramma = np.array(self.sismogramma)
        # print("ho caricato hdf5", time.perf_counter()-start)
        datd = dd.read_csv(percorsocsv, dtype={"trace_P_arrival_sample": int})
        # non metto engine, assume missinng etc perchè questi selezionati sembrano buoni
        # print("ho letto csv", time.perf_counter()-start)
        self.metadata = {}
        for key in datd:
            self.metadata[key] = np.array(datd[key])
            # print("ho caricato la key ", key, time.perf_counter() - start)
        self.centrato = self.metadata["centrato"][1]
        self.demeaned = self.metadata["demeaned"][1]
        # print(self.sismogramma.shape, len(self.sismogramma))

    def to_txt(self, txt_data, txt_metadata):
        # self.acquisisci_new(percorsohdf5, percorsocsv, col_tot=col_tot, nomi_selezionati=nomi_selezionati)
        # print("\n\nVA BENE?", self.sismogramma)
        np.savetxt(txt_data, self.sismogramma, fmt='%.5e')  # warning, ma fuonziona ok
        metadata_txt = pd.DataFrame.from_dict(self.metadata)
        metadata_txt.to_csv(txt_metadata, index=False, sep='\t')
        # df.to_csv(r'c:\data\pandas.txt', header=None, index=None, sep='\t', mode='a')

    def leggi_classi_txt(self, percorsoclassi):
        """
        Legge le classi SOM a cui sono assegnate le tracce
        self.classi[i] = k --> la traccia i-esima appartiene alla classe k-esima della som
        ATTENTO mi serve sapere da quale dataset provengono: Posso ricavare solo l'indice della traccia
        """
        self.classi = []
        with open(percorsoclassi, 'r') as f:
            for line in f:
                if line:  # avoid blank lines
                    self.classi.append(int(float(line.strip())))

    def ricava_indici_classi(self, classi_da_selezionare: list, vettore_indici: list):
        """
         metto in vettore_indici gli indici delle tracce che appartengono ad una delle classi elencate in
         classi_da_selezionare

         ATTENTO vettore indici deve essere fornito come LISTA vuota
        """
        if len(self.sismogramma) != len(self.classi):
            stringa = "#"
            for _ in range(300):
                stringa = stringa + "#"
            warnings.warn("\n" + stringa + "\nATTENTO, CLASSI e SISMOGRAMMA LUNGHEZZA DIFFERENTE"
                                           "continuo senza centrare nulla\n" + stringa)
            print("classi length = ", len(self.classi), "sismogrammma len = ", len(self.sismogramma))
            input()
            return 1

        for i in range(len(self.classi)):
            for j in classi_da_selezionare:
                if self.classi[i] == j:
                    vettore_indici.append(i)
        # print("vettore indici interno funzione", vettore_indici)

    def calcola_media(self, nome_medie):
        """
            Calcola le medie (prima dell'onda o tutta traccia) max per ciascun sismogramma
            e le  salva nel file nome_medie
        """
        medie = []
        medie_rumore = []
        massimo_abs = []
        for i in range(len(self.sismogramma)):
            massimo_abs.append(max(self.sismogramma[i].max(), -self.sismogramma[i].min()))
            medie.append(np.mean(self.sismogramma[i]))
            if self.centrato:
                medie_rumore.append(np.mean(self.sismogramma[i][:len(self.sismogramma[i]) // 2 - 5]))
            else:
                medie_rumore.append(np.mean(self.sismogramma[i][:self.metadata["trace_P_arrival_sample"][i] - 10]))

        pd_mean_max = pd.DataFrame({"media_totale": medie, "media_rumore": medie_rumore, "max": massimo_abs})
        pd_mean_max.to_excel(nome_medie + ".xlsx", index=False)

    def finestra(self, semiampiezza=0):
        """
            taglia e se necessario centra la finestra
            semiampiezza: numero di samples (0.01s) es 100 per finestra di 2 sec
            se non è già centrato/tagliato, quando finestra è big -> return 1 e non modifica sellf.sismogramma
            se  è già centrato/tagliato, quando finestra è big    -> non modifica sellf.sismogramma
        """
        sismogramma = [0 for _ in range(len(self.sismogramma))]

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
                    input()
                    return 1

            self.sismogramma = np.array(sismogramma)

            self.centrato = True

    def demean(self, metodo):
        """
            scrive su file media e media_rumore diviso il valore massimo per ciascun sismogramma
            metodo totale -> toglie tutta la media
            metodo rumore -> toglie la media calcolata su valori prima dell'arrivo dell'onda P
        """
        # self.calcola_media("con_media")
        if metodo == "totale":
            self.demeaned = "totale"
            for i in range(len(self.sismogramma)):
                self.sismogramma[i] = self.sismogramma[i] - np.mean(self.sismogramma[i])

        if metodo == "rumore":
            self.demeaned = "rumore"
            if self.centrato:
                for i in range(len(self.sismogramma)):
                    lung = len(self.sismogramma[0])
                    self.sismogramma[i] = self.sismogramma[i] - np.mean(self.sismogramma[i][:lung//2-10])
            else:
                for i in range(len(self.sismogramma)):
                    self.sismogramma[i] = self.sismogramma[i] - \
                                          np.mean(self.sismogramma[i][:self.metadata["trace_P_arrival_sample"][i] - 10])

        if metodo != "rumore" and metodo != "totale":
            print("attento, metodo demean sbagliato\n rumore o totale? (NON SO SE FUNZIONa ORA)")
            metodo = input()
            if metodo == "rumore" or metodo == "totale":
                self.demean(metodo)
        # self.calcola_media("senza_media")

    def normalizza(self, soglia=20):
        """
        # TODO implementa giusta normalizzazione (da decidere)
        Metodo 1, prova
        """
        lung_traccia = len(self.sismogramma[0])
        self.sismogramma = self.sismogramma * 1.0                 # ATTENTISSIMO, altrimenti ho np array di interi
        for i in range(len(self.sismogramma)):
            max_rumore = np.max(self.sismogramma[i][0:lung_traccia//2-5])
            min_rumore = np.min(self.sismogramma[i][0:lung_traccia//2-5])
            if soglia*max(max_rumore, -min_rumore) == 0:
                print("questa è la traccia in cui divido per 0\t", i)
            self.sismogramma[i] = self.sismogramma[i]/(soglia*max(max_rumore, -min_rumore))
            for j in range(lung_traccia):
                self.sismogramma[i][j] = min(self.sismogramma[i][j], 1)
                self.sismogramma[i][j] = max(self.sismogramma[i][j], -1)

            if i % 1000 == 0:
                print("normalizzo, sto alla ", i)

    def elimina_tacce_indici(self, vettore_indici: list):
        """
        vettore_indici è la lista degli indici (in accordo con file csv) da eliminare
        info proviene da leggi_classi_txt, che legge le classi della SOM o
        se voglio eliminare tracce in altra maniera selezionate
         a = np.delete(a,[2,1],axis=0) elimina le righe 2 e 1 del vettore
        """
        # print("lemetadat",  type(self.metadata))
        self.sismogramma = np.delete(self.sismogramma, vettore_indici, axis=0)
        for key in self.metadata:   # controlla tutorial_dizionari, funziona ok #TODO verifica che è buono
            self.metadata[key] = np.array(np.delete(self.metadata[key], vettore_indici, axis=0))
            self.metadata[key] = list(self.metadata[key])

    def elimina_tacce_nomi(self, vettore_nomi: list):
        """
        vettore_nomi è la lista dei nomi delle tracce da eliminare
        """
        indici_eliminare = []
        for j in range(len(vettore_nomi)):
            for i in range(len(self.sismogramma)):
                if vettore_nomi[j] == self.metadata["trace_name"][i]:
                    indici_eliminare.append(i)
                    break               # TODO mi sembra buono (tutorial_break)

        self.elimina_tacce_indici(vettore_indici=indici_eliminare)

    def plotta(self, visualizza, semiampiezza=None, namepng=None, percosro_cartellla='C:/Users/GioCar/Desktop/Tesi_5'):
        """
        visualizza:                 lista di indici delle tracce da visualizzare
        semiampiezza:               della finestra da visualizzare
        namepng:                    se è passato diventa il nome del file in cui salvo i plot
        percorso_cartella           salva in 'C:/Users/GioCar/Desktop/Tesi_5/'+namepng se non è passato

        # TODO migliora algoritmo, rendilo più legibbile
        """
        if len(self.sismogramma) < len(visualizza):
            print("lunghezza sismogramma < sismogrammi da visualizzare")
            return 1
        if namepng is not None:
            nome_cartella = percosro_cartellla + "/" + namepng
            os.mkdir(nome_cartella)

        if self.centrato:
            if semiampiezza is None or semiampiezza > len(self.sismogramma[0])//2:
                semiampiezza = len(self.sismogramma[0])//2
            for i in visualizza:
                lung = len(self.sismogramma[0])
                plt.plot(range(2*semiampiezza), self.sismogramma[i][lung//2 - semiampiezza:
                                                                    lung//2 + semiampiezza])
                plt.axvline(x=semiampiezza, c="r", ls="--",lw=1)
                plt.axhline(y=0, color='k', ls = 'dashed', lw=1)
                stringa = ""
                for key in self.metadata:
                    if key != "centrato" and key != "demeaned":
                        stringa = stringa + str(self.metadata[key][i]) + " "
                # stringa = stringa + str(self.indice_csv[i])
                plt.title(stringa)
                if namepng is None:
                    plt.show()
                else:
                    nome_cartella = percosro_cartellla + "/" + namepng
                    plt.savefig(nome_cartella + "/" + namepng + "_" + str(i))
                    plt.clf()

        else:  # NON E' CENTRATO o TAGLIATO
            semiampiezza_ori = semiampiezza
            for i in visualizza:
                if semiampiezza_ori is None or semiampiezza_ori > self.metadata["trace_P_arrival_sample"][i]:
                    semiampiezza = self.metadata["trace_P_arrival_sample"][i]-1
                else:
                    semiampiezza = semiampiezza_ori
                # FIXME
                """
                da errore, non so perchè
                Dataset_1.plotta(visualizza=140, semiampiezza=1000, namepng="new/vedi135")
                per sismogramma non centrato ed semiampiezza superiore tracearrivalp
                """
                plt.plot(range(2*semiampiezza),
                         self.sismogramma[i][self.metadata["trace_P_arrival_sample"][i] - semiampiezza:
                                             self.metadata["trace_P_arrival_sample"][i] + semiampiezza])
                plt.axvline(x=semiampiezza, c="r", ls="--", lw="1")
                plt.axhline(y=0, color='k', ls='--', lw=1)
                stringa = ""
                for key in self.metadata:
                    if key != "centrato" and key != "demeaned":
                        stringa = stringa + str(self.metadata[key][i]) + " "
                # stringa = stringa + str(self.indice_csv[i])
                plt.title(stringa)
                if namepng is None:
                    plt.show()
                else:
                    nome_cartella = percosro_cartellla + "/" + namepng
                    plt.savefig(nome_cartella + "/" + namepng + "_" + str(i))
                    plt.clf()





if __name__ == "main":
    print("ciao")

    # csvin = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/metadata/metadata_Instance_events_10k.csv'
    # hdf5in = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/data/Instance_events_counts_10k.hdf5'
# csvout = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/metadata_Instance_events_selected_Polarity_Velocimeter.csv'
    # hdf5out = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/data_selected_Polarity_Velocimeter.hdf5'
    # txt_data = "C:/Users/GioCar/Desktop/Tesi_5/txt_tracce.txt"
    # txt_metadata = "C:/Users/GioCar/Desktop/Tesi_5/txt_metadata.txt"
    # coltot = ["trace_name", "station_channels", "trace_P_arrival_sample", "trace_polarity",
    #           "trace_P_uncertainty_s", "source_magnitude", "source_magnitude_type"]
    # nomi = "Selezionati.csv"
    #
    # #####trace_name,station_channels needed
    # Dataset_1 = Classe_Dataset()
    # Dataset_1.to_txt(hdf5in, csvin, ["trace_name", "station_channels", "trace_P_arrival_sample",
    #                                  "trace_polarity", "source_magnitude"], txt_data, txt_metadata)
    # Dataset_1.leggi_custom_dataset(hdf5out,csvout)
    # Dataset_1.crea_custom_dataset(hdf5out,csvout,coltot=coltot)
    # Dataset_1.acquisisci_new(percorsohdf5=hdf5in, percorsocsv=csvin, coltot=coltot, nomi_selezionati=nomi)
    # Dataset_1.plotta(visualizza=30, namepng="Dataset_counts")
    # Dataset_1.acquisisci_old(percorsohdf5=hdf5in, percorsocsv=csvin, coltot=coltot, percorso_nomi=nomi)
# Dataset_1.plotta(visualizza=range(5), namepng="/home/silvia/Desktop/Figure_Large_Custom_dataset/Custom_Large_dataset")
    # Dataset_1.Finestra(1000000)
    # Dataset_1.plotta(50, semiampiezza=100, namepng="prova")

    # hdf5in = '/home/silvia/Desktop/Instance_Data/Uno/data_selected_Polarity_Velocimeter.hdf5'
    # csvin = '/home/silvia/Desktop/Instance_Data/Uno/metadata_Instance_events_selected_Polarity_Velocimeter.csv'
    # medieprima = '/home/silvia/Desktop/Instance_Data/Due/medieprima'
    # mediedopo = '/home/silvia/Desktop/Instance_Data/Due/mediedopo'
    #
    # coltot = ["trace_name", "station_channels", "trace_P_arrival_sample", "trace_polarity",
    #           "trace_P_uncertainty_s", "source_magnitude", "source_magnitude_type"]
    # hdf5out = '/home/silvia/Desktop/Instance_Data/Due/data_selected_Polarity_Velocimeter_4s.hdf5'
    # csvout = '/home/silvia/Desktop/Instance_Data/Due/metadata_Instance_events_selected_Polarity_Velocimeter_4s.csv'
    #
    # Dataset_1 = Classe_Dataset()
    # Dataset_1.leggi_custom_dataset(percorsohdf5=hdf5out, percorsocsv=csvout)
    #
    # Dataset_1.plotta(visualizza=[83911, 26410, 22696], semiampiezza=200,
    #                  namepng="/home/silvia/Desktop/Instance_Data/Due/visione1")
    # csvin = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/metadata/metadata_Instance_events_10k.csv'
    # hdf5in = 'C:/Users/GioCar/Desktop/Tesi_5/Simple_dataset/data/Instance_events_counts_10k.hdf5'
    # coltot = ["trace_name", "station_channels", "trace_P_arrival_sample", "trace_polarity",
    #           "trace_P_uncertainty_s", "source_magnitude", "source_magnitude_type"]
    # classiup = 'C:/Users/GioCar/Desktop/Tesi_5/SOM/3classes_up.txt'
    # classidown = 'C:/Users/GioCar/Desktop/Tesi_5/SOM/4classes_down.txt'
    #
    # nomi_down = "Selezionati_down.csv"
    #
    # Dataset_d = ClasseDataset()
    # Dataset_d.acquisisci_old(hdf5in, csvin, coltot, nomi_down)
    # Dataset_d.plotta([1, 2, 3, 4], semiampiezza=150, namepng="funziona")

    # classi = [0 for i in range(len(Dataset_1.sismogramma))]
    # for i in [0, 1, 2, 3]:
    #     classi[i] = 1
    # Dataset_1.plotta(range(5))
    # Dataset_1.elimina_tacce(range(4))
    # Dataset_1.plotta(range(5))
