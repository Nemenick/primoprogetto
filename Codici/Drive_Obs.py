import obspy
from obspy import read

cosa = read("/home/silvia/Desktop/Pollino/20101015010010_M1.9/20101015005956.CUC.HHZ.sac")  # legge tutti i sac non zippati
print(cosa, type(cosa))
print("\n", cosa[0], "\n", type(cosa[0]))
print("\ndata", cosa[0].data, "\ntype", type(cosa[0].data))
print("\nstats", cosa[0].stats, "\ntype", type(cosa[0].stats))
print("\nkeys", cosa[0].stats.keys)
cosa.plot()