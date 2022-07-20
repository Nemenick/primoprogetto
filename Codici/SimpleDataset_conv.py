import numpy as np
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from Classe_sismogramma_v2 import Classe_Dataset

Dati = Classe_Dataset()
csvin = 'C:/Users/GioCar/Desktop/Simple_dataset/metadata/metadata_Instance_events_10k.csv'
hdf5in = 'C:/Users/GioCar/Desktop/Simple_dataset/data/Instance_events_counts_10k.hdf5'
coltot = ["trace_name", "station_channels", "trace_P_arrival_sample", "trace_polarity",
          "trace_P_uncertainty_s", "source_magnitude", "source_magnitude_type"]
nomi = "Selezionati.csv"
Dati.acquisisci_old(hdf5in, csvin, coltot=coltot, percorso_nomi=nomi)

x_train = np.zeros((len(Dati.sismogramma)*2, 260))
for i in range(len(Dati.sismogramma)):
    x_train[i] = Dati.sismogramma[i][Dati.metadata["trace_P_arrival_sample"][i] - 130:
                                     Dati.metadata["trace_P_arrival_sample"][i] + 130]
    x_train[i+len(Dati.sismogramma)] = -Dati.sismogramma[i][Dati.metadata["trace_P_arrival_sample"][i] - 130:
                                        Dati.metadata["trace_P_arrival_sample"][i] + 130]

print(type(x_train), x_train.shape)
y_train = np.array([Dati.metadata["trace_polarity"][i] == "positive" for i in range(len(Dati.sismogramma))] +
                   [1-(Dati.metadata["trace_polarity"][i] == "positive") for i in range(len(Dati.sismogramma))])
print("\nsomma ytrain", np.sum(y_train))    # OK si trova

(x_val, y_val) = (x_train[0:len(x_train)//10], y_train[0:len(x_train)//10])
(x_train, y_train) = (x_train[len(x_train)//10:len(x_train)], y_train[len(x_train)//10:len(x_train)])


model = keras.models.Sequential([
    Conv1D(64, 3, input_shape=(len(x_train[0]),  1), activation="relu"),           # FIXME attento (len,1) o (len,) ???
    MaxPooling1D(2),
    Conv1D(128, 3, activation="relu"),
    MaxPooling1D(2),
    Conv1D(32, 3, activation="relu"),
    MaxPooling1D(2),
    Flatten(),
    Dense(50, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['accuracy']
)

epoche = 50
storia = model.fit(x_train, y_train, batch_size=16, epochs=epoche, validation_data=(x_val, y_val))
model.save("Simple_data_conv_1.0.hdf5")
print("\n\nQUI\n", storia.history,)
print(storia.history.keys())
loss_train = storia.history["loss"]
loss_val = storia.history["val_loss"]
acc_train = storia.history["accuracy"]
acc_val = storia.history["val_accuracy"]

plt.plot(range(1, epoche+1), loss_train, label="loss_train")
plt.plot(range(1, epoche+1), loss_val, label="loss_val")
plt.legend()
plt.show()

plt.plot(range(1, epoche+1), acc_train, label="acc_train")
plt.plot(range(1, epoche+1), acc_val, label="acc_val")
plt.legend()
plt.show()

# predizione = model.evaluate(x_test, y_test)
#
# print(len(predizione), y_test.shape, type(predizione), type(y_test))
# print("predict", predizione)
