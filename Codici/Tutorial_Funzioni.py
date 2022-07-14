import numpy as np


def fun(a):
    a = a + 3
    print("ida", id(a))
    print(a)

v = np.array([1, 2])
print("idv", id(v))

fun(v)
print(v)

b = 3
print("idb, id3", id(b), id(3))
