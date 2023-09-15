from tensorflow import keras
from keras.layers import Dense

pat_tent = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/'
tentativo = 52

model = keras.models.load_model(pat_tent+str(tentativo)+'/Tentativo_'+str(tentativo)+'.hdf5')
model.summary()

newLAy = Dense(1, input_shape = (50,), activation=None)
newLAy.build((None, 50))
a = model.layers[-1].get_weights()
newLAy.set_weights(a)