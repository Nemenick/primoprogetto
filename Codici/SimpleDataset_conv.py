from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from Classe_sismogramma_v2 import Classe_Dataset

Dati = Classe_Dataset()

Dati