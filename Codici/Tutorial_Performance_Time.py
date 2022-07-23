import h5py
import numpy as np
import time
import dask.dataframe as dd

Nprove= 100

"""# TODO append vs acess previouslu created list
lung = 10**6
start = time.perf_counter()
for _ in range(Nprove):
    a = [0 for __ in range(lung)]
    for i in range(len(a)):
        a[i] = i
print("tempo previously created", time.perf_counter()-start)

start = time.perf_counter()
for _ in range(Nprove):
    a = []
    for i in range(lung):
        a.append(i)
print("tempo append", time.perf_counter()-start)"""

"""# start = time.perf_counter()
# for _ in range(Nprove):
#     datd = dd.read_csv('C:/Users/GioCar/Desktop/Simple_dataset/metadata/metadata_Instance_events_10k.csv', usecols=["source_id", "trace_P_arrival_sample", "trace_polarity"])
#     #print(np.array(datd["source_id"]))
#     Parrival_times = np.array(datd["trace_P_arrival_sample"])
#     Ppolarity = np.array(datd["trace_polarity"])
# end = time.perf_counter()
# print("lettura CSV", end - start)
#
# filehdf5 = h5py.File('C:/Users/GioCar/Desktop/Simple_dataset/data/Instance_events_counts_10k.hdf5', 'r')
# dataset = filehdf5.get("data")
# nomidata = list(dataset.keys())
"""

"""# Todo Mi sono salvato i nomi di tutti i dataset
# start1 = time.perf_counter()                                               # PRIMO METODO
# for _ in range(Nprove):
#     sismogramma = []
#     for i in range(len(nomidata)):
#         sismogramma.append(dataset.get(nomidata[i]))
#     sismogramma = np.array(sismogramma)
# end1 = time.perf_counter()
# print("primo metodo", end1-start1)
#
# start2 = time.perf_counter()                                                # SECONDO METODO
# for _ in range(Nprove):
#     sismogramma = [0 for i in range(10000)]
#     for i in range(len(nomidata)):
#         sismogramma[i] = dataset.get(nomidata[i])
#     sismogramma = np.array(sismogramma)
# end2 = time.perf_counter()
# print("secondo metodo", end2-start2, type(end2))

# start3 = time.perf_counter()                                               # TERZO METODO
# for _ in range(Nprove):
#     sismogramma = []
#     for i in range(len(nomidata)):
#         if(Ppolarity[i] != 'undecidable'):
#             sismogramma.append(dataset.get(nomidata[i]))
#             sismogramma[-1] = sismogramma[-1][2]
#     sismogramma = np.array(sismogramma)
# end3 = time.perf_counter()
# print("terzo metodo", end3 - start3, sismogramma.shape)
# print(sismogramma[2])"""

"""
# Todo comparison while , for scorrere una lista
# a = [0 for i in range(10**7)]
# print("ciao")
# start = time.perf_counter()
# for j in range(Nprove):
#     for i in range(len(a)):
#         a[i] = 0
# end = time.perf_counter()
# print("tempo for", end-start)
#
# start = time.perf_counter()
# for j in range(Nprove):
#     i = 0
#     while i < len(a):
#         a[i] = 0
#         i = i + 1
# end = time.perf_counter()
# print("tempo while", end-start)
"""