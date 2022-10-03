import obspy
import numpy as np
from obspy import read
import datetime
from matplotlib import pyplot as plt
cosa = read("/home/silvia/Desktop/Pollino/*/*Z.sac", format="SAC")  # legge tutti i sac non zippati
# cosa = read("/home/silvia/Desktop/Pollino/20101015010010_M1.9/20101015005956.CUC.HHZ.sac")
print(cosa, type(cosa))
print("\n", cosa[0], "\n", type(cosa[0]))
print("\ndata", cosa[0].data, "\ntype", type(cosa[0].data))
print("\nstats", cosa[0].stats, "\ntype", type(cosa[0].stats))
print("\nkeys", cosa[0].stats.keys)
for traccia in cosa:
    print(traccia.stats['npts'], traccia.stats['sampling_rate'], traccia.stats['channel'], traccia.stats['delta'])

#TODO np.concatenate (()) DOPPIA PARENTESI!

# TODO per polarity
# cosa[0].stats['sac']['ka']

# TODO per arrival sample
"""
start = cosa[0].stats['starttime']
print(start.datetime, type(start.datetime), type(start))
data_list = ['nzyear', 'nzjday', 'nzhour', 'nzmin', 'nzsec', 'nzmsec']
d = []
for i in data_list:
    print(cosa[0].stats['sac'][i])
    d.append(cosa[0].stats['sac'][i])
reference = obspy.UTCDateTime(year=d[0], julday=d[1], hour=d[2], minute=d[3], second=d[4], microsecond=d[5]*1000)

print(reference.datetime, type(reference.datetime), type(reference))
arrival = (reference.datetime-start.datetime).total_seconds()+cosa[0].stats['sac']['a']
print(arrival)
# cosa.plot()
plt.plot(cosa[0].data)
plt.axvline(x=int(arrival*100), c="r", ls="--", lw="1")
plt.show()

np_cosa = np.array(cosa[0].data)
print(np_cosa, type(np_cosa), type(np_cosa[0]))
plt.plot(np_cosa)
plt.axvline(x=int(arrival*100), c="r", ls="--", lw="1")
plt.show()
"""
