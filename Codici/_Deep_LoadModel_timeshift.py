import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from Classe_sismogramma_v3 import ClasseDataset


def dividi_train_test_val(estremi_test: list, estremi_val: list, semi_amp: int, dati: ClasseDataset):
    """
    :param estremi_val:     estremi della validation set, [lat_min, lat_max, lon_min, lon_max]
    :param estremi_test:    idem di e_val
    :param semi_amp:        Semiampiezza della traccia da considerare
    :param dati:            dataset da suddividere
    :return:                xytrain, xytest, xyval (come np.array), dati_test,dati_val come ClasseDataset
    """

    lung = len(dati.sismogramma[0])  # lunghezza traccia
    (xval, yval, xtest, ytest) = ([], [], [], [])
    indici_test = []
    indici_val = []
    for k in range(len(dati.sismogramma)):
        if estremi_test[0] < dati.metadata['source_latitude_deg'][k] < estremi_test[1] and estremi_test[2] \
                < dati.metadata['source_longitude_deg'][k] < estremi_test[3]:
            indici_test.append(k)                   # li farò eliminare dal trainset
            xtest.append(dati.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp])
            if dati.metadata["trace_polarity"][k] == "positive":
                ytest.append(1)
            elif dati.metadata["trace_polarity"][k] == "negative":
                ytest.append(0)
        if estremi_val[0] < dati.metadata['source_latitude_deg'][k] < estremi_val[1] and estremi_val[2] \
                < dati.metadata['source_longitude_deg'][k] < estremi_val[3]:
            indici_val.append(k)                   # li farò eliminare dal trainset
            xval.append(dati.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp])
            if dati.metadata["trace_polarity"][k] == "positive":
                yval.append(1)
            elif dati.metadata["trace_polarity"][k] == "negative":
                yval.append(0)
    (xval, yval, xtest, ytest) = (np.array(xval), np.array(yval), np.array(xtest), np.array(ytest))
    print("\n\nPrint da funzione dividi.. : lunghezze val,test", len(xval), len(yval), len(xtest), len(ytest), "\n\n")

    dati_test = dati.seleziona_indici(indici_test)
    dati_val = dati.seleziona_indici(indici_val)

    indici_test_val = indici_test + indici_val
    dati.elimina_tacce_indici(indici_test_val)
    sample_train = len(dati.sismogramma)
    xtrain = np.zeros((sample_train * 2, semi_amp * 2))
    for k in range(sample_train):
        xtrain[k] = dati.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp]
        xtrain[k + sample_train] = -dati.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp]
    ytrain = np.array([dati.metadata["trace_polarity"][_] == "positive" for _ in range(sample_train)] +
                      [1 - (dati.metadata["trace_polarity"][_] == "positive") for _ in range(sample_train)])
    ytrain = ytrain + 0
    return xtrain, ytrain, xtest, ytest, xval, yval, dati_test, dati_val


e_test = [43, 45, 9.5, 11.8]
e_val = [37.5, 38.5, 14.5, 16]              # TODO cambia qui e controlla se non esistono già le cartelle

hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s_Normalizzate.hdf5'
csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s_Normalizzate.csv'

# hdf5in = '/home/silvia/Desktop/Pollino/Pollino_100Hz_data.hdf5'
# csvin = '/home/silvia/Desktop/Pollino/Pollino_100Hz_metadata.csv'

Dati = ClasseDataset()
Dati.leggi_custom_dataset(hdf5in, csvin)
print("N_sismogrammi", len(Dati.sismogramma), "N_polarità", len(Dati.metadata["trace_polarity"]))
# Dati.elimina_tacce_indici([124709])

semiampiezza = 80       # TODO
x_train, y_train, x_test, y_test, x_val, y_val, Dati_test, Dati_val = dividi_train_test_val(e_test, e_val,
                                                                                            semiampiezza, Dati)
test_sample = len(Dati_test.sismogramma)
lung = len(Dati.sismogramma[0])
x = np.zeros((len(Dati_test.sismogramma)+len(Dati_val.sismogramma), semiampiezza*2))
y = np.array([Dati_test.metadata["trace_polarity"][i] == "positive" for i in range(len(Dati_test.sismogramma))] +
             [Dati_val.metadata["trace_polarity"][i] == "positive" for i in range(len(Dati_val.sismogramma))])
y = y + 0
pat_tent = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/'
tentativi = [39]
# kern_sizs = [3, 5, 7, 9]
time_shifts = [(i-40) for i in range(81)]
predizioni = [[[], [], []] for i in range(len(tentativi))]
for k in range(len(tentativi)):
    tentativo = tentativi[k]
    # kern_siz = kern_sizs[k]
    model = keras.models.load_model(pat_tent+str(tentativo)+'/Tentativo_'+str(tentativo)+'.hdf5')  # TODO
    model.summary()

    for time_shift in time_shifts:
        for i in range(len(Dati_test.sismogramma)):
            x[i] = Dati_test.sismogramma[i][lung // 2 - semiampiezza + time_shift:lung // 2 + semiampiezza + time_shift]
        for i in range(len(Dati_val.sismogramma)):
            x[i+test_sample] = Dati_val.sismogramma[i][lung // 2 - semiampiezza + time_shift:lung // 2 + semiampiezza + time_shift]

        predizione = model.evaluate(x, y, batch_size=1024)
        predizioni[k][0].append(time_shift)
        predizioni[k][1].append(predizione[0])
        predizioni[k][2].append(predizione[1])
        print("predict num", time_shift, predizione)

    plt.plot(predizioni[k][0], predizioni[k][1], label="Loss_Agumented_timeshift")  # TODO
plt.legend()
# plt.show()
plt.savefig(pat_tent + "/" + "Loss_Agumented_long_timeshift")
plt.clf()

for k in range(len(tentativi)):
    tentativo = tentativi[k]
    # kern_siz = kern_sizs[k]
    plt.plot(predizioni[k][0], predizioni[k][2], label="Accuracy_Agumented_timeshift")  # TODO
plt.legend()
# plt.show()
plt.axhline(0.5, color='k', ls='dashed', lw=1)
plt.axhline(0.75, color='k', ls='dashed', lw=1)
plt.savefig(pat_tent + "/" + "Accuracy_Agumented_long_timeshift")
plt.clf()

# yp = model.predict(x)
# print(y, len(y), "\n", yp, len(yp))
# yp_ok = []
# for i in yp:
#     yp_ok.append(i[0])
# yp_ok = np.array(yp_ok)
# delta_y = np.abs(y-yp_ok)
#
# tracce_previsione_errata = []
# tracce_previsione_incerta = []
# for i in range(len(delta_y)):
#     if delta_y[i] > 0.5:
#         tracce_previsione_errata.append(i)
#     if 0.2 < delta_y[i] < 0.5:
#         tracce_previsione_incerta.append(i)
# Dati.plotta(tracce_previsione_errata, 130, "figure_previsione_errata_80_tentativo9", "/home/silvia/Desktop/Pollino")
# Dati.plotta(tracce_previsione_incerta, 130, "figure_previsione_incerta_80_tentativo9", "/home/silvia/Desktop/Pollino")  # TODO
# dizio = {"traccia": Dati.metadata["trace_name"], "y_a_Mano": y, "y_predict": yp_ok, "delta": delta_y}
# datapandas = pd.DataFrame.from_dict(dizio)
# datapandas.to_csv('/home/silvia/Desktop/Predizioni_Pollino_semiampiezza_80_tentativo9.csv', index=False)                # TODO
#
# print(type(yp))