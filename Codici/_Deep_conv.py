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

semiampiezza = 11
# Dati.plotta(range(200),semiampiezza,"normalizzati",'/home/silvia/Desktop')
lung = len(Dati.sismogramma[0])     # lunghezza traccia

# TODO Agumentation
# """
x_train = np.zeros((len(Dati.sismogramma)*2, semiampiezza*2))
for i in range(len(Dati.sismogramma)):
    x_train[i] = Dati.sismogramma[i][lung//2 - semiampiezza:lung//2 + semiampiezza]
    x_train[i+len(Dati.sismogramma)] = -Dati.sismogramma[i][lung//2 - semiampiezza:lung//2 + semiampiezza]
y_train = np.array([Dati.metadata["trace_polarity"][i] == "positive" for i in range(len(Dati.sismogramma))] +
                   [1-(Dati.metadata["trace_polarity"][i] == "positive") for i in range(len(Dati.sismogramma))])
y_train = y_train + 0
(x_val, y_val) = (x_train[0:len(x_train)//10], y_train[0:len(x_train)//10])
(x_train, y_train) = (x_train[len(x_train)//10:len(x_train)], y_train[len(x_train)//10:len(x_train)])
# """

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
model.summary()

# Inizio il train

epoche = 6
start = time.perf_counter()
storia = model.fit(x_train, y_train, batch_size=16, epochs=epoche, validation_data=(x_val, y_val))
# vedi validation come evolve durante la stessa epoca
print("\n\n\nTEMPOO per ", epoche, "epoche: ", time.perf_counter()-start, "\n\n\n")
model.save("Tentativo_8.hdf5")
print("\n\nControlla qui\n", storia.history)
print(storia.history.keys())

loss_train = storia.history["loss"]
loss_val = storia.history["val_loss"]
acc_train = storia.history["accuracy"]
acc_val = storia.history["val_accuracy"]

plt.plot(range(1, epoche+1), acc_train, label="acc_train")
plt.plot(range(1, epoche+1), acc_val, label="acc_val")
plt.legend()
plt.savefig("accuracy8")
plt.clf()


plt.yscale("log")
plt.plot(range(1, epoche+1), loss_train, label="loss_train")
plt.plot(range(1, epoche+1), loss_val, label="loss_val")
plt.legend()
plt.savefig("loss8")
plt.clf()
file = open("_Dettagli_8.txt", "w")
dettagli = "semiampiezza = " + str(semiampiezza) + \
           "\ndati normalizzati con primo metodo " + hdf5in +\
           "\nepoche = " + str(epoche)
file.write(dettagli)
file.close()
"""
N_test = len(x_val)
yp = model.predict(x_val[0:N_test])
yp_new = [val[0] for val in yp]
print(y_train, "\n", yp_new)
dizio = {"y_INGV": y_val[0:N_test], "y_predict": yp_new}
datapandas = pd.DataFrame.from_dict(dizio)
datapandas.to_csv('/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/Predizioni_nonAgumented.csv', index=False)
"""
# predizione = model.evaluate(x_test, y_test)
#
# print(len(predizione), y_test.shape, type(predizione), type(y_test))
# print("predict", predizione)
