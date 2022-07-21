
print("\n\n\nINIZIO")
devices = tf.config.list_physical_devices('GPU')
print(len(devices))
tf.test.is_built_with_cuda()
print(tf.test.is_built_with_cuda())
print("\n\n\nFINE")