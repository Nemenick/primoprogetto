import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from Classe_sismogramma_v3 import ClasseDataset

Dati = ClasseDataset()

csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s_Normalizzate.csv'   # percorso di dove sono contenuti i metadata
hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s_Normalizzate.hdf5'       # percorso di Dove sono contenute le tracce
"""
csvin = 'C:/Users/GioCar/Desktop/Tesi_5/metadata_Velocimeter_Buone_normalizzate1_4s.csv'
hdf5in = 'C:/Users/GioCar/Desktop/Tesi_5/data_Velocimeter_Buone_normalizzate_4s.hdf5'
"""


Dati.leggi_custom_dataset(hdf5in, csvin)  # Leggo il dataset
Dati.elimina_tacce_indici([124709])       # FIXME attento questa traccia è nan per buone_normalizzate_Instance

semiampiezza = 80
batchs = 512                               # TODO
sample_train = len(Dati.sismogramma)                     # num di tracce da dare come train (il resto è validation)
tentativo = "12"

# Dati.plotta(range(200),semiampiezza,"normalizzati",'/home/silvia/Desktop')
lung = len(Dati.sismogramma[0])     # lunghezza traccia

# TODO Agumentation
# """
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
# TODO uso Pollino come validation
# sample_val = len(Dati.sismogramma) - sample_train
# x_val = np.zeros((sample_val, semiampiezza*2))
# for i in range(sample_val):
#     x_val[i] = Dati.sismogramma[i+sample_train][lung//2 - semiampiezza:lung//2 + semiampiezza]
# y_val = np.array(
#     [Dati.metadata["trace_polarity"][i+sample_train] == "positive" for i in range(sample_val)]                )
# y_val = y_val + 0
# # (x_val, y_val) = (x_train[0:len(x_train)//10], y_train[0:len(x_train)//10])
# # (x_train, y_train) = (x_train[len(x_train)//10:len(x_train)], y_train[len(x_train)//10:len(x_train)])
# # """


# TODO NON Agumentation
"""
x_train = np.zeros((len(Dati.sismogramma), semiampiezza*2))
for i in range(len(Dati.sismogramma)):
    x_train[i] = Dati.sismogramma[i][lung//2 - semiampiezza:lung//2 + semiampiezza]
y_train = np.array([Dati.metadata["trace_polarity"][i] == "positive" for i in range(len(Dati.sismogramma))])
y_train = y_train + 0
(x_val, y_val) = (x_train[0:len(x_train)//10], y_train[0:len(x_train)//10])
(x_train, y_train) = (x_train[len(x_train)//10:len(x_train)], y_train[len(x_train)//10:len(x_train)])
"""

"""for j in range(5):
    plt.plot(x_train[j+10], label=str(y_train[j+10]))
plt.legend()
plt.show()"""

# input shape : 1D convolutions and recurrent layers use(batch_size, sequence_length, features)
# batch size omitted ... (len(timeseries),1 (channels)) funziona
# Creo la mia rete deep con i layer
#  TODO Prima rete
"""
rete = 1
model = keras.models.Sequential([
    Conv1D(64, 3, input_shape=(len(x_train[0]), 1), activation="relu"),
    MaxPooling1D(2),
    Conv1D(128, 3, activation="relu"),
    MaxPooling1D(2),
    Conv1D(32, 3, activation="relu"),
    MaxPooling1D(2),                         # rete di esempio, giusto per imparare a maneggiare le deep
    Flatten(),                               # TODO implementa batch normalizzation, dropout, fit_generator, custom loss
    Dense(50, activation="relu"),            # TODO prendi spunto da reti Ross, Uchide, Hara per migliore architettura
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
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['accuracy']
)
# """

model.summary()

# Inizio il train

epoche = 350
start = time.perf_counter()
storia = model.fit(x_train, y_train, batch_size=batchs, epochs=epoche, validation_data=(x_val, y_val))
# vedi validation come evolve durante la stessa epoca
print("\n\n\nTEMPOO per ", epoche, "epoche: ", time.perf_counter()-start, "\n\n\n")
model.save("Tentativo_"+tentativo+".hdf5")
print("\n\nControlla qui\n", storia.history)
print(storia.history.keys())

loss_train = storia.history["loss"]
loss_val = storia.history["val_loss"]
acc_train = storia.history["accuracy"]
acc_val = storia.history["val_accuracy"]

plt.plot(range(1, epoche+1), acc_train, label="acc_train")
plt.plot(range(1, epoche+1), acc_val, label="acc_val")
plt.legend()
plt.savefig("accuracy_"+tentativo)
plt.clf()


plt.yscale("log")
plt.plot(range(1, epoche+1), loss_train, label="loss_train")
plt.plot(range(1, epoche+1), loss_val, label="loss_val")
plt.legend()
plt.savefig("loss_"+tentativo)
plt.clf()

file = open("_Dettagli_"+tentativo+".txt", "w")
dettagli = "Rete numero " + str(rete) + \
            "\nbatchsize = " + str(batchs) + \
            "\nsemiampiezza = " + str(semiampiezza) + \
            "\ndati normalizzati con primo metodo " + hdf5in +\
            "\nepoche = " + str(epoche) +\
            "\nsample_train = " + str(sample_train) +\
            "\nIn questo train la validation sono i dai del Pollino, il train è tutto instance dataset"
file.write(dettagli)
file.close()

dizio = {"loss_train": loss_train, "loss_val": loss_val, "acc_train": acc_train, "acc_val": acc_val}
data_pandas = pd.DataFrame.from_dict(dizio)
data_pandas.to_csv('/home/silvia/Documents/GitHub/primoprogetto/Codici/Risultati_' + tentativo + '.cvs')
# TODO predict
# """

yp = model.predict(x_val)
print(y_val, len(y_val), "\n", yp, len(yp))
yp_ok = []
for i in yp:
    yp_ok.append(i[0])
yp_ok = np.array(yp_ok)
delta_y = np.abs(y_val-yp_ok)

tracce_previsione_errata = []
tracce_previsione_incerta = []
for i in range(len(delta_y)):
    if delta_y[i] > 0.5:
        tracce_previsione_errata.append(i)
    if 0.2 < delta_y[i] < 0.5:
        tracce_previsione_incerta.append(i)
Dati_validation.plotta(tracce_previsione_errata, 130, "figure_previsione_errata_tentativo_"+str(tentativo), "/home/silvia/Desktop/Pollino")
Dati_validation.plotta(tracce_previsione_incerta, 130, "figure_previsione_incerta_tentativo_"+str(tentativo), "/home/silvia/Desktop/Pollino")  # TODO
dizio_val = {"traccia": Dati_validation.metadata["trace_name"], "y_a_Mano": y_val, "y_predict": yp_ok, "delta": delta_y}
datapandas = pd.DataFrame.from_dict(dizio_val)
datapandas.to_csv('/home/silvia/Documents/GitHub/primoprogetto/Codici/'
                  'Predizioni_Pollino_semiampiezza_tentativo_'+str(tentativo)+'.csv', index=False)                # TODO
# """

# predizione = model.evaluate(x_test, y_test)
#
# print(len(predizione), y_test.shape, type(predizione), type(y_test))
# print("predict", predizione)
