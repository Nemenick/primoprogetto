import time
import numpy as np
import pandas as pd
import random
# from keras.utils.np_utils import to_categorical
# import tensorflow as tf
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"          # TODO ATTENTO !
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping
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

def dividi_train_test_val_shift2(estremi_test: list, estremi_val: list, semi_amp: int, dati: ClasseDataset, timeshift=0):
    """
    :param estremi_val:     estremi della validation set, [lat_min, lat_max, lon_min, lon_max]
    :param estremi_test:    idem di e_val
    :param semi_amp:        Semiampiezza della traccia da considerare
    :param dati:            dataset da suddividere
    :param timeshift:       effettuo un timeshift con dist. uniforme in [0,t] e [-t,0] per le sole tracce del train
    :return:                xytrain, xytest, xyval (come np.array), dati_test,dati_val come ClasseDataset
    
    train 4-uplicato in taglia.
    ora xtrain sara: [dati_normali[0:tot], dati_norm_shifted+[pari], dati_norm_shifted-[pari]]+
                     [dati_flip[0:tot], dati_flip_shifted+[pari], dati_flip_shifted-[pari]]
                     non pari ma quelli per cui k>sample_train/2 !!!
    """

    lung = len(dati.sismogramma[0])  # lunghezza traccia
    (xval, yval, xtest, ytest) = ([], [], [], [])
    indici_test = []
    indici_val = []
    for k in range(len(dati.sismogramma)):
        if estremi_test[0] < dati.metadata['source_latitude_deg'][k] < estremi_test[1] and estremi_test[2] \
                < dati.metadata['source_longitude_deg'][k] < estremi_test[3]:
            indici_test.append(k)                   # li farò eliminare dal trainset
            shift = random.randint(-timeshift, timeshift) * 0  # TODO
            xtest.append(dati.sismogramma[k][lung // 2 - semi_amp + shift:lung // 2 + semi_amp + shift])
            if dati.metadata["trace_polarity"][k] == "positive":
                ytest.append(1)
            elif dati.metadata["trace_polarity"][k] == "negative":
                ytest.append(0)
        if estremi_val[0] < dati.metadata['source_latitude_deg'][k] < estremi_val[1] and estremi_val[2] \
                < dati.metadata['source_longitude_deg'][k] < estremi_val[3]:
            indici_val.append(k)                   # li farò eliminare dal trainset
            shift = random.randint(-timeshift, timeshift) * 0  # TODO
            xval.append(dati.sismogramma[k][lung // 2 - semi_amp + shift:lung // 2 + semi_amp + shift])
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
    if sample_train % 2 == 1:
        sample_train -= 1
    xtrain = np.zeros((sample_train * 4, semi_amp * 2))
    for k in range(sample_train):
        xtrain[k] = dati.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp]
        xtrain[k + 2*sample_train] = -dati.sismogramma[k][lung // 2 - semi_amp:lung // 2 + semi_amp]
        if k > sample_train/2:
            shift1 = random.randint(1, timeshift)      # shift alla traccia normale positivo
            shift_1 = random.randint(1, timeshift)     # ""    alla traccia normale negativo
            shift_11 = random.randint(1, timeshift)    # ""    alla traccia flipped positivo
            shift_1_1 = random.randint(1, timeshift)   # ""    alla traccia flipped negativo
            # cast in int serve solo perchè come indicizzazione vuole un int SICURO. In realtà sample_train % 2 == 0
            # vedi fig per capire xtrain
            xtrain[int(k- sample_train/2 + sample_train)] = dati.sismogramma[k][lung // 2 - semi_amp + shift1:lung // 2 + semi_amp + shift1]
            xtrain[int(k- sample_train/2 + 3*sample_train/2)] = dati.sismogramma[k][lung // 2 - semi_amp - shift_1:lung // 2 + semi_amp - shift_1]
            xtrain[int(k -sample_train/2 + 7*sample_train/2)] = -dati.sismogramma[k][lung // 2 - semi_amp + shift_11:lung // 2 + semi_amp + shift_11]
            xtrain[int(k-sample_train/2 + 3 * sample_train)] = -dati.sismogramma[k][lung // 2 - semi_amp - shift_1_1:lung // 2 + semi_amp - shift_1_1]

    ytrain = \
        np.array([dati.metadata["trace_polarity"][_] == "positive" for _ in range(sample_train)] +
                 [dati.metadata["trace_polarity"][_] == "positive" for _ in range(int(sample_train/2), sample_train)]*2
                 + [1 - (dati.metadata["trace_polarity"][_] == "positive") for _ in range(sample_train)] +
                 [1 - (dati.metadata["trace_polarity"][_] == "positive") for _ in range(int(sample_train/2),
                                                                                        sample_train)]*2)
    ytrain = ytrain + 0
    return xtrain, ytrain, xtest, ytest, xval, yval, dati_test, dati_val



hdf5in = '/home/silvia/Desktop/Data/Instance_Data/Tre_4s/data_Velocimeter_4s_Normalizzate_New1-1.hdf5'
csvin = '/home/silvia/Desktop/Data/Instance_Data/Tre_4s/metadata_Velocimeter_4s_Normalizzate_New1-1.csv'

Dati = ClasseDataset()
Dati.leggi_custom_dataset(hdf5in, csvin)  # Leggo il dataset

e_test = [43, 45, 9.5, 11.8]
e_val = [37.5, 38.5, 14.5, 16]              # TODO cambia qui e controlla se non esistono già le cartelle
n_train = 8     # number of training to mean (number of networks)
tentativo = "16"
path_tentativi = f'/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/More_{tentativo}'

os.mkdir(path_tentativi)
for i in range(n_train):
    os.mkdir(path_tentativi + "/" + str(i))

semiampiezza = 80
epoche = 85
batchs = 512                                # TODO CAMBIA parametri
pazienza = 10
drop = 0.5
shift_samples = 10
versione_shift = "2"

if shift_samples == 0:
    x_train, y_train, x_test, y_test, x_val, y_val, Dati_test, Dati_val = dividi_train_test_val(e_test, e_val,
                                                                                                semiampiezza, Dati)
else:
    x_train, y_train, x_test, y_test, x_val, y_val, Dati_test, Dati_val = dividi_train_test_val_shift2(e_test, e_val,
                                                                                            semiampiezza, Dati,
                                                                                            timeshift=shift_samples)
# input shape : 1D convolutions and recurrent layers use(batch_size, sequence_length, features)
# batch size omitted ... (len(timeseries),1 (channels)) funziona
# Creo la mia rete deep con i layer


list_acc_test = []
list_acc_val = []

for tent in range(n_train):

    # epsilon = 10**(-2)  # TODO cambia (al prossimo....)
    # print('\n\tepsilon = ', epsilon)
    momento = 0.8
    print('\n\tmomento = ', momento)

    #  TODO prima rete
    """
    rete = 1
    model = keras.models.Sequential([
        Conv1D(64, 3, input_shape=(len(x_train[0]), 1), activation="relu"),
        MaxPooling1D(2),
        Dropout(drop),
        Conv1D(128, 3, activation="relu"),
        MaxPooling1D(2),
        Conv1D(32, 3, activation="relu"),
        MaxPooling1D(2),                     
        Flatten(),                           
        Dense(50, activation="relu"),        
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=optimizers.SGD(momentum=momento),  # TODO CAMBIA
        # optimizer=optimizers.Adam(epsilon=epsilon),
        loss="binary_crossentropy",
        metrics=['accuracy']
    )
    """
    #  TODO Seconda rete
    
    rete = 2
    model = keras.models.Sequential([
        Conv1D(32, 5, input_shape=(len(x_train[0]), 1), activation="relu", padding="same"),
        Dropout(drop),
        Conv1D(64, 4, activation="relu"),
        MaxPooling1D(2),
        Conv1D(128, 3, activation="relu"),
        MaxPooling1D(2),
        Conv1D(256, 5, activation="relu", padding="same"),
        Dropout(drop),
        Conv1D(128, 3, activation="relu"),
        MaxPooling1D(2),
        Flatten(),
        Dense(50, activation="softsign"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.SGD(momentum=momento),  # TODO CAMBIA
        # optimizer=optimizers.Adam(epsilon=epsilon),
        loss="binary_crossentropy",
        metrics=['accuracy']
    )
    

    model.summary() 
    print(f"######################Inizio il {int(tent)+1}o training ######################")

    # Inizio il train

    start = time.perf_counter()
    storia = model.fit(x_train, y_train,
                       batch_size=batchs,
                       epochs=epoche,
                       validation_data=(x_val, y_val),
                       callbacks=EarlyStopping(patience=pazienza,  restore_best_weights=True))
    print("\n\n\nTEMPOO per ", epoche, "epoche: ", time.perf_counter()-start, "\n\n\n")
    model.save(path_tentativi + "/" + str(tent) + f"/Tentativo_More_{tentativo}_"+str(tent)+".hdf5")
    print("\n\nControlla qui\n", storia.history)
    print(storia.history.keys())

    loss_train = storia.history["loss"]
    loss_val = storia.history["val_loss"]
    acc_train = storia.history["accuracy"]
    acc_val = storia.history["val_accuracy"]

    plt.plot(range(len(acc_train)), acc_train, label="acc_train")
    plt.plot(range(len(acc_val)), acc_val, label="acc_val")
    plt.legend()
    plt.savefig(path_tentativi + "/" + str(tent) + f"/accuracy_More_{tentativo}_"+str(tent))
    plt.clf()

    plt.yscale("log")
    plt.plot(range(len(loss_train)), loss_train, label="loss_train")
    plt.plot(range(len(loss_val)), loss_val, label="loss_val")
    plt.legend()
    plt.savefig(path_tentativi + "/" + str(tent) + f"/loss_More_{tentativo}_"+str(tent))
    plt.clf()

    dizio = {"loss_train": loss_train, "loss_val": loss_val, "acc_train": acc_train, "acc_val": acc_val}
    data_pandas = pd.DataFrame.from_dict(dizio)
    data_pandas.to_csv(path_tentativi + "/" + str(tent) + f'/Storia_train_More_{tentativo}_' + str(tent) + '.csv', index=False)

    # TODO predict
    # """
    model = keras.models.load_model(path_tentativi + "/" + str(tent) + f"/Tentativo_More_{tentativo}_"+str(tent)+".hdf5")
    yp_test = model.predict(x_test)
    # print(y_test, len(y_test), "\n", yp_test, len(yp_test))
    yp_ok_test = []
    for i in yp_test:
        yp_ok_test.append(i[0])
    yp_ok_test = np.array(yp_ok_test)
    delta_y_test = np.abs(y_test - yp_ok_test)

    yp_val = model.predict(x_val)
    # print(y_val, len(y_val), "\n", yp_val, len(yp_val))
    yp_ok_val = []
    for i in yp_val:
        yp_ok_val.append(i[0])
    yp_ok_val = np.array(yp_ok_val)
    delta_y_val = np.abs(y_val - yp_ok_val)

    acc_fin_test = 0
    acc_fin_val = 0
    for delta in delta_y_val:
        if abs(delta) < 0.5:
            acc_fin_val += 1
    acc_fin_val = acc_fin_val / len(delta_y_val)
    for delta in delta_y_test:
        if abs(delta) < 0.5:
            acc_fin_test += 1
    acc_fin_test = acc_fin_test / len(delta_y_test)

    list_acc_val.append(acc_fin_val)
    list_acc_test.append(acc_fin_test)

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
    datapandas_test.to_csv(path_tentativi + "/" + str(tent) +
                           f"/Predizioni_test_tentativo_More_{tentativo}_" + str(tent) + ".csv", index=False)
    datapandas_val.to_csv(path_tentativi + "/" + str(tent) +
                          f"/Predizioni_val_tentativo_More_{tentativo}_" + str(tent) + ".csv", index=False)
    # """


lista_media_test, lista_std_test = ["" for i in range(len(list_acc_test))], ["" for i in range(len(list_acc_test))]
lista_media_val, lista_std_val = ["" for i in range(len(list_acc_test))], ["" for i in range(len(list_acc_test))]
lista_media_test[0] = np.mean(list_acc_test)
lista_std_test[0] = np.std(list_acc_test)
lista_media_val[0] = np.mean(list_acc_val)
lista_std_val[0] = np.std(list_acc_val)
dizio_acc = {"Acc_val": list_acc_val, "Acc_test": list_acc_test, "Mean_val": lista_media_val, "Std_val": lista_std_val, "Mean_test": lista_media_test, "Std_test": lista_std_test}

datapandas_acc = pd.DataFrame.from_dict(dizio_acc)
datapandas_acc.to_csv(path_tentativi +
                      "/Accuratezze_vari_train" + ".csv", index=False)

file = open(path_tentativi  + "/_Dettagli_More_"+str(tentativo)+".txt", "w")
# TODO Cambia i dettagli
dettagli =  f"""Rete numero  {rete}
            batchsize = {batchs}
            semiampiezza = {semiampiezza} 
            dati normalizzati con primo metodo {hdf5in}
            sample_train = {len(x_train)/2}
            In questo train train,test,val sono instance
            coordinate test = {e_test}) con {len(x_test)} dati di test 
            coordinate val = {e_val}) con {len(x_val)} dati di val 
            Optimizer: SGD con momento = {momento}) 
            Early_stopping con patiente = {pazienza})  restore_best_weights = True
            DROPOUT ({drop}) dopo 1o e 4o cpnv\n
            HO FATTO {n_train} train diversi, devo controllare medie 
            conv -> max -> drop -> conv -> max -> conv -> max -> dense -> dense
            timeshift nel training set = {shift_samples}
            versione_shift = {versione_shift}
            """

file.write(dettagli)
file.close()
# predizione = model.evaluate(x_test, y_test)
#
# print(len(predizione), y_test.shape, type(predizione), type(y_test))
# print("predict", predizione)


