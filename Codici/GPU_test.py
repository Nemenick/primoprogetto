import tensorflow as tf

print("\n\n\nINIZIO")
tf.test.is_gpu_available()
print("aaaa")
print(tf.debugging.set_log_device_placement(True))
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print("c è questo",c)


devices = tf.config.list_physical_devices()
print("devices",devices)
print("build cuda",tf.test.is_built_with_cuda())
print("devices 2.0",tf.config.list_physical_devices(device_type=None))
print("FINE\n\n\n")
