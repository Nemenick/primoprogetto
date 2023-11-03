import tensorflow as tf

print("\n\n\nINIZIO")
tf.config.list_physical_devices('GPU')

print("\n\naaaa")
print(tf.debugging.set_log_device_placement(True))
print("\n\n")
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print("c è questo tensore", c)


devices = tf.config.list_physical_devices()
print("\n\n")
print("devices", devices)
print("\n\n")
print("build cuda", tf.test.is_built_with_cuda())
print("FINE\n\n\n")
