import tensorflow as tf
import numpy as np




# [var.name for var in tape.watched_variables()] printa ultime variabili su cui sto facendo gradiente

# TODO gradeinti di modello
"""
# ATTENZIONE non sono sicuro di aver capito come sono gli shape di Conv1D... (batch, len_timeseries, channels) ????

layer = tf.keras.layers.Conv1D(1,3, activation='relu')
flat =tf.keras.layers.Flatten()
layer1 = tf.keras.layers.Dense(2, activation='relu')
layer2 = tf.keras.layers.Dense(1, activation='relu')

x = tf.constant([[[1.], [2.], [3.],[4],[5],[6],[7],[8],[9]]])
with tf.GradientTape() as tape:
  # Forward pass
  tape.watch(x)
  y = layer(x)
  y1 = flat(y)
  y2 = layer1(y1)
  print(y2.shape)
  z = layer2(y2)
  loss = tf.reduce_mean(z**2)
# Calculate gradients with respect to x
grad = tape.gradient(loss, x)
print("outshape",z)
print(loss,grad, "FUNONZIA")"""




# TODO WATCH... permette di settare quale variabile sto usando per fare gradiente. Di default le costanti non sono watched...
"""layer0 = tf.keras.layers.Dense(3, activation='relu')
x_0 = layer0(x)
a = [np.diag([1,1,1]),np.zeros((3))]
layer0.set_weights(a)
print("Layer0 eccolo qui\n", layer0.weights, "\nqui finisce\n")"""