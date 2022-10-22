import time
import numpy as np
import pandas as pd
# from keras.utils.np_utils import to_categorical
# import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from Classe_sismogramma_v3 import ClasseDataset
import os


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


csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s_Normalizzate.csv'
# percorso di dove sono contenuti i metadata
hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s_Normalizzate.hdf5'
# percorso di Dove sono contenute le tracce
"""
csvin = 'C:/Users/GioCar/Desktop/Tesi_5/metadata_Velocimeter_Buone_normalizzate_4s.csv'
hdf5in = 'C:/Users/GioCar/Desktop/Tesi_5/data_Velocimeter_Buone_normalizzate_4s.hdf5'
"""

Dati = ClasseDataset()
Dati.leggi_custom_dataset(hdf5in, csvin)  # Leggo il dataset

e_val = [43, 45, 9.5, 11.8]
e_test = [37.5, 38.5, 14.5, 16]              # TODO cambia qui e controlla se non esistono già le cartelle
tentativi = [30]
# epsilons = [10**(-5), 0.001, 0.1]
path_tentativi = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi'
for tentativo in tentativi:
    os.mkdir(path_tentativi + "/" + str(tentativo))

semiampiezza = 80
epoche = 300
batchs = 512

x_train, y_train, x_test, y_test, x_val, y_val, Dati_test, Dati_val = dividi_train_test_val(e_test, e_val,
                                                                                            semiampiezza, Dati)

# input shape : 1D convolutions and recurrent layers use(batch_size, sequence_length, features)
# batch size omitted ... (len(timeseries),1 (channels)) funziona
# Creo la mia rete deep con i layer


for tentativo in tentativi:
    epsilon = 10**(-3)  # TODO cambia (al prossimo....)
    print("\n\tepsilon = ", epsilon)

    #  TODO Prima rete
    """
    rete = 1
    model = keras.models.Sequential([
        Conv1D(64, 3, input_shape=(len(x_train[0]), 1), activation="relu"),
        MaxPooling1D(2),
        Conv1D(128, 3, activation="relu"),
        MaxPooling1D(2),
        Conv1D(32, 3, activation="relu"),
        MaxPooling1D(2),                     # rete di esempio, giusto per imparare a maneggiare le deep
        Flatten(),                           # TODO implementa batch normalizzation, dropout, fit_generator, custom loss
        Dense(50, activation="relu"),        # TODO prendi spunto da reti Ross, Uchide, Hara per migliore architettura
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=['accuracy']
    )
    """

    #  TODO Seconda rete
    # """
    rete = 2
    model = keras.models.Sequential([
        Conv1D(32, 5, input_shape=(len(x_train[0]), 1), activation="relu", padding="same"),
        Conv1D(64, 4, activation="relu"),
        MaxPooling1D(2),

        Conv1D(128, 3, activation="relu"),
        MaxPooling1D(2),

        Conv1D(256, 5, activation="relu", padding="same"),
        Conv1D(128, 3, activation="relu"),
        MaxPooling1D(2),

        Flatten(),
        Dense(50, activation="softsign"),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        # optimizer=optimizers.SGD(momentum=momento),  # TODO CAMBIA
        optimizer=optimizers.Adam(epsilon=epsilon),
        loss="binary_crossentropy",
        metrics=['accuracy']
    )
    # """

    model.summary()

    # Inizio il train

    start = time.perf_counter()
    storia = model.fit(x_train, y_train,
                       batch_size=batchs,
                       epochs=epoche,
                       validation_data=(x_val, y_val),
                       )  # callbacks=EarlyStopping(patience=3,  restore_best_weights=True)
    print("\n\n\nTEMPOO per ", epoche, "epoche: ", time.perf_counter()-start, "\n\n\n")
    model.save(path_tentativi + "/" + str(tentativo) + "/Tentativo_"+str(tentativo)+".hdf5")
    print("\n\nControlla qui\n", storia.history)
    print(storia.history.keys())

    loss_train = storia.history["loss"]
    loss_val = storia.history["val_loss"]
    acc_train = storia.history["accuracy"]
    acc_val = storia.history["val_accuracy"]

    plt.plot(range(len(acc_train)), acc_train, label="acc_train")
    plt.plot(range(len(acc_val)), acc_val, label="acc_val")
    plt.legend()
    plt.savefig(path_tentativi + "/" + str(tentativo) + "/accuracy_"+str(tentativo))
    plt.clf()

    plt.yscale("log")
    plt.plot(range(len(loss_train)), loss_train, label="loss_train")
    plt.plot(range(len(loss_val)), loss_val, label="loss_val")
    plt.legend()
    plt.savefig(path_tentativi + "/" + str(tentativo) + "/loss_"+str(tentativo))
    plt.clf()

    file = open(path_tentativi + "/" + str(tentativo) + "/_Dettagli_"+str(tentativo)+".txt", "w")
    # TODO Cambia i dettagli
    dettagli = "Rete numero " + str(rete) + \
               "\nbatchsize = " + str(batchs) +\
               "\nsemiampiezza = " + str(semiampiezza) +\
               "\ndati normalizzati con primo metodo " + hdf5in +\
               "\nepoche = " + str(epoche) +\
               "\nsample_train = " + str(len(x_train)/2) +\
               "\nIn questo train train,test,val sono instance" + \
               "\ncoordinate test = " + str(e_test) + "con "+str(len(x_test))+" dati di test" + \
               "\ncoordinate val = " + str(e_val) + "con "+str(len(x_val))+" dati di val" + \
               "\nOptimizer: ADMA con epsilon = " + str(epsilon) + \
               "\nOra ho scambiato test e val, cerco di spiegare perchè val meglio di train(almeno inizialmente)" + \
               " Confronta con 22"
    # Early_stopping    con    patiente = 3, restore_best_weights = True
    file.write(dettagli)
    file.close()

    dizio = {"loss_train": loss_train, "loss_val": loss_val, "acc_train": acc_train, "acc_val": acc_val}
    data_pandas = pd.DataFrame.from_dict(dizio)
    data_pandas.to_csv(path_tentativi + "/" + str(tentativo) + '/Storia_train_' + str(tentativo) + '.csv', index=False)

    # TODO predict
    # """
    yp_test = model.predict(x_test)
    print(y_test, len(y_test), "\n", yp_test, len(yp_test))
    yp_ok_test = []
    for i in yp_test:
        yp_ok_test.append(i[0])
    yp_ok_test = np.array(yp_ok_test)
    delta_y_test = np.abs(y_test - yp_ok_test)

    yp_val = model.predict(x_val)
    print(y_val, len(y_val), "\n", yp_val, len(yp_val))
    yp_ok_val = []
    for i in yp_val:
        yp_ok_val.append(i[0])
    yp_ok_val = np.array(yp_ok_val)
    delta_y_val = np.abs(y_val - yp_ok_val)

    # tracce_previsione_errata = []
    # tracce_previsione_incerta = []
    # for i in range(len(delta_y)):
    #     if delta_y[i] > 0.5:
    #         tracce_previsione_errata.append(i)
    #     if 0.2 < delta_y[i] < 0.5:
    #         tracce_previsione_incerta.append(i)
    # Dati_validation.plotta(tracce_previsione_errata, 130, "figure_previsione_errata_tentativo_"+str(tentativo),
    # "/home/silvia/Desktop/Pollino")
    # Dati_validation.plotta(tracce_previsione_incerta, 130, "figure_previsione_incerta_tentativo_"+str(tentativo),
    # "/home/silvia/Desktop/Pollino")  # TODO

    dizio_val = {"traccia_val": Dati_val.metadata["trace_name"], "y_Mano_val": y_val, "y_predict_val": yp_ok_val,
                 "delta_val": delta_y_val}
    dizio_test = {"traccia_test": Dati_test.metadata["trace_name"], "y_Mano_test": y_test,
                  "y_predict_test": yp_ok_test, "delta_test": delta_y_test}
    datapandas_test = pd.DataFrame.from_dict(dizio_test)
    datapandas_val = pd.DataFrame.from_dict(dizio_val)
    datapandas_test.to_csv(path_tentativi + "/" + str(tentativo) +
                           "/Predizioni_test_tentativo_" + str(tentativo) + ".csv", index=False)
    datapandas_val.to_csv(path_tentativi + "/" + str(tentativo) +
                          "/Predizioni_val_tentativo_" + str(tentativo) + ".csv", index=False)
    # """

# predizione = model.evaluate(x_test, y_test)
#
# print(len(predizione), y_test.shape, type(predizione), type(y_test))
# print("predict", predizione)

# TODO vecchia divisione dataset
"""
sample_train = len(Dati.sismogramma)                     # num di tracce da dare come train (il resto è validation)
# tentativo = "18"

# Dati.plotta(range(200),semiampiezza,"normalizzati",'/home/silvia/Desktop')
lung = len(Dati.sismogramma[0])     # lunghezza traccia

# TODO Agumentation

x_train = np.zeros((sample_train*2, semiampiezza*2))
for i in range(sample_train):
    x_train[i] = Dati.sismogramma[i][lung//2 - semiampiezza:lung//2 + semiampiezza]
    x_train[i+sample_train] = -Dati.sismogramma[i][lung//2 - semiampiezza:lung//2 + semiampiezza]
y_train = np.array([Dati.metadata["trace_polarity"][i] == "positive" for i in range(sample_train)] +
                   [1-(Dati.metadata["trace_polarity"][i] == "positive") for i in range(sample_train)])
y_train = y_train + 0

Dati_validation = ClasseDataset()
hdf5val = '/home/silvia/Desktop/Pollino/Pollino_100Hz_data.hdf5'
csvval = '/home/silvia/Desktop/Pollino/Pollino_100Hz_metadata.csv'
Dati_validation.leggi_custom_dataset(hdf5val, csvval)
lung_val = len(Dati_validation.sismogramma[0])
x_val = np.zeros((len(Dati_validation.sismogramma), semiampiezza*2))
for i in range(len(Dati_validation.sismogramma)):
    x_val[i] = Dati_validation.sismogramma[i][lung // 2 - semiampiezza:lung // 2 + semiampiezza]
y_val = np.array([Dati_validation.metadata["trace_polarity"][i] == "positive"
                  for i in range(len(Dati_validation.sismogramma))])
y_val = y_val + 0
#  uso Pollino come validation
# sample_val = len(Dati.sismogramma) - sample_train
# x_val = np.zeros((sample_val, semiampiezza*2))
# for i in range(sample_val):
#     x_val[i] = Dati.sismogramma[i+sample_train][lung//2 - semiampiezza:lung//2 + semiampiezza]
# y_val = np.array(
#     [Dati.metadata["trace_polarity"][i+sample_train] == "positive" for i in range(sample_val)]                )
# y_val = y_val + 0
# # (x_val, y_val) = (x_train[0:len(x_train)//10], y_train[0:len(x_train)//10])
# # (x_train, y_train) = (x_train[len(x_train)//10:len(x_train)], y_train[len(x_train)//10:len(x_train)])



#  NON Agumentation

x_train = np.zeros((len(Dati.sismogramma), semiampiezza*2))
for i in range(len(Dati.sismogramma)):
    x_train[i] = Dati.sismogramma[i][lung//2 - semiampiezza:lung//2 + semiampiezza]
y_train = np.array([Dati.metadata["trace_polarity"][i] == "positive" for i in range(len(Dati.sismogramma))])
y_train = y_train + 0
(x_val, y_val) = (x_train[0:len(x_train)//10], y_train[0:len(x_train)//10])
(x_train, y_train) = (x_train[len(x_train)//10:len(x_train)], y_train[len(x_train)//10:len(x_train)])
"""
