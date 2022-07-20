from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt


# 60000 input images
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
# print(x_train[1])
(y_train, y_test) = (to_categorical(y_train), to_categorical(y_test))
(x_val, y_val) = (x_train[0:len(x_train)//10], y_train[0:len(x_train)//10])
(x_train, y_train) = (x_train[len(x_train)//10:len(x_train)], y_train[len(x_train)//10:len(x_train)])

model = keras.models.Sequential([
    Conv2D(16, (3, 3), input_shape=(28, 28, 1), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(20, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=['accuracy']
)
epoche = 5
storia = model.fit(x_train, y_train, batch_size=16, epochs=epoche, validation_data=(x_val, y_val))
model.save("MNIST_conv.hdf5")
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

predizione = model.evaluate(x_test, y_test)

print(len(predizione), y_test.shape, type(predizione), type(y_test))
print("predict", predizione)