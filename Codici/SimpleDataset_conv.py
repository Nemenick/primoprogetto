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

x = np.zeros((len(Dati.sismogramma), 200))
for i in range(len(Dati.sismogramma)):
    x[i] = Dati.sismogramma[i][Dati.metadata["trace_P_arrival_sample"][i] - 100:
                                              Dati.metadata["trace_P_arrival_sample"][i] + 100]
print(type(x), x.shape)

model = keras.models.Sequential([
    Conv1D(16, 3, input_shape=(len(Dati.sismogramma[0]),  1), activation="relu"),
    MaxPooling1D(2),
    Conv1D(32, 3, activation="relu"),
    MaxPooling1D(2),
    Flatten(),
    Dense(20, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['accuracy']
)
