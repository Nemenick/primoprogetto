import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from matplotlib import pyplot as plt
from Classe_sismogramma_v3 import ClasseDataset
# mi servono 10 giga disponibili di RAM (minimo)
###################### FAI gradiente su ultimo ma prima di fare funzione di attivazione!#####################


def create_identity(x):
    vinput_shape = (None, 160, 1)
    identity = Conv1D(filters=1, kernel_size=1, activation=None, input_shape=vinput_shape[1:])
    x_0 = x
    x_0 = np.expand_dims(x_0, axis=0)
    x_0 = np.expand_dims(x_0, axis=-1)
    aia = identity(x_0)                
    identity.set_weights([np.array([[[1.0]]]), np.array([0.0])])
    return identity

def calc_single_saliency(model, input_k, identity):
    with tf.GradientTape() as tape:
        x_0 = input_k
        x_0 = np.expand_dims(x_0, axis=0)
        x_0 = np.expand_dims(x_0, axis=-1)
        seq = [x_0]

        seq.append(identity(x_0))

        for layer_i in model.layers:
            seq.append(layer_i(seq[-1]))
                                  
        grad = tape.gradient(seq[-1],seq[1])
    grad = np.array(grad)
    grad = grad[0]
    return (grad, seq[-1])

hdf5_predicting = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s_Normalizzate_New1-1.hdf5'
csv_predicting = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s_Normalizzate_New1-1.csv'
Data_predicting = ClasseDataset()
Data_predicting.leggi_custom_dataset(hdf5_predicting, csv_predicting)
lung = len(Data_predicting.sismogramma[0])
semi_amp = 80
pat_tent = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/'

tentativo = 79
time_shift = 10                                                                  
figs_name = f"/home/silvia/Desktop/Immagini/Saliency\
/Saliency_InstancePol_tent{tentativo}_shift{time_shift}"                    # Da cambiare tra i run
saveplot = True

# TODO predict Instance Test (genero dati input rete)
# """
estremi_test = [43, 45, 9.5, 11.8]
xtest = []
y_test_true = []
test_indici = []
for k in range(len(Data_predicting.sismogramma)):
    if estremi_test[0] < Data_predicting.metadata['source_latitude_deg'][k] < estremi_test[1] and estremi_test[2] \
            < Data_predicting.metadata['source_longitude_deg'][k] < estremi_test[3]:
        xtest.append(Data_predicting.sismogramma[k][lung // 2 - semi_amp + time_shift:lung // 2 + semi_amp + time_shift])

        test_indici.append(k)  # TODO salvo posizioni tracce di test
        if Data_predicting.metadata["trace_polarity"][k] == "positive":
            y_test_true.append(1)
        elif Data_predicting.metadata["trace_polarity"][k] == "negative":
            y_test_true.append(0)
Data_predicting = Data_predicting.seleziona_indici(test_indici)
# """

# TODO predict other than Instance polarity
"""
xtest = np.zeros((sample_train, semi_amp * 2))
for k in range(sample_train):
    xtest[k] = Data_predicting.sismogramma[k][lung // 2 - semi_amp + time_shift:lung // 2 + semi_amp + time_shift]
y_test_true = np.array([Data_predicting.metadata["trace_polarity"][_] == "positive" for _ in range(sample_train)])
y_test_true = y_test_true + 0
"""
xtest = np.array(xtest)
print("XSHAPEEEEEEEEEEEE", xtest.shape)

model = keras.models.load_model(pat_tent+str(tentativo)+'/Tentativo_'+str(tentativo)+'.hdf5')
model.summary()


# TODO Da qui inizio a calcolare i gradienti per generare la Saliency map

# Creo layer identità... mi serve perchè non riuscivo a fare gradiente rispetto input (non serve a niente concettualmente)
identi_layer = create_identity(xtest[0])

Saliency_stacked = np.zeros(160)
Saliency_stacked_abs = np.zeros(160)
for input_k in xtest[100:200]:

    grad_k, output_k = calc_single_saliency(model, input_k, identi_layer)
    #print("caio grad",grad.shape)
    Saliency_single = np.sum(grad_k, axis = 1)                                                      # faccio solamente un reshape da (160,1) a (160,)
    Saliency_single = Saliency_single/ np.max([np.max(Saliency_single),-np.min(Saliency_single)])   # normalizzo per il max assoluto
    Saliency_stacked += Saliency_single                                                             # Saliency map mediata
    Saliency_stacked_abs += np.abs(Saliency_single)                                                 # Saliency map mediata (in valore assoluto)

if saveplot:
    plt.plot(Saliency_stacked)
    plt.savefig(figs_name+"stack.jpg")
    plt.clf()
    plt.plot(Saliency_stacked_abs)
    plt.savefig(figs_name+"stack_abs_NEW.jpg")



# Per tirocinio:
# Fai salienci di: tent 52 e tentativo 79, di timeshift tra -20 e 20, passo di 2 e fai una piccola relazione su quello che succede
# (cosa stanno facendo le reti? su che zona si concentrano, limiti utilizzabilità reti?)