import obspy
import numpy as np
from obspy import read
import datetime
import pandas as pd
from matplotlib import pyplot as plt


# path = "/home/silvia/Desktop/Pollino/*/*Z.sac"
path = "/home/silvia/Desktop/viste_buone/*/*Z.sac"
path_pandas = "/home/silvia/Desktop/polarity.csv"
tracce_sac = read(path, format="SAC")  # legge tutti i sac non zippati
# cosa = read("/home/silvia/Desktop/Pollino/20101015010010_M1.9/20101015005956.CUC.HHZ.sac")
print(tracce_sac, type(tracce_sac))
print("\n", tracce_sac[0], "\n", type(tracce_sac[0]))
print("\ndata", tracce_sac[0].data, "\ntype", type(tracce_sac[0].data))
print("\nstats", tracce_sac[0].stats, "\ntype", type(tracce_sac[0].stats))
print("\nkeys", tracce_sac[0].stats.keys)
# for traccia in tracce_sac:
#     print(traccia.stats['npts'], traccia.stats['sampling_rate'], traccia.stats['channel'], traccia.stats['delta'])


trace_name = []                         # ok
station_channels = []                   # ok
trace_P_arrival_sample = []             # ok
trace_polarity = []
trace_P_uncertainty_s = []              # ok
source_magnitude = []                   # ok
source_magnitude_type = []              # ok
sampling_rate = []                      # ok

data_list = ['nzyear', 'nzjday', 'nzhour', 'nzmin', 'nzsec', 'nzmsec']
for i in range(len(tracce_sac)):

    # TODO np.concatenate (()) DOPPIA PARENTESI!
    # per append Classedataset.data
    # TODO per polarity
    pol = tracce_sac[i].stats['sac']['ka']
    if pol.find("U") >= 0:
        trace_polarity.append("positive")
    elif pol.find("D") >= 0:
        trace_polarity.append("negative")
    else:
        trace_polarity.append("undecidable")
    # TODO per arrival sample
    """
    start = tracce_sac[i].stats['starttime']
    # print(start.datetime, type(start.datetime), type(start))

    d = []
    for j in data_list:
        # print(tracce_sac[i].stats['sac'][j])
        d.append(tracce_sac[i].stats['sac'][j])
    reference = obspy.UTCDateTime(year=d[0], julday=d[1], hour=d[2], minute=d[3], second=d[4], microsecond=d[5]*1000)

    # print(reference.datetime, type(reference.datetime), type(reference))
    arrival = (reference.datetime-start.datetime).total_seconds() + tracce_sac[i].stats['sac']['a']
    if i % 200 == 0:
        print(arrival, tracce_sac[i].stats['npts'], " sto alla ", i)
    trace_P_arrival_sample.append(int(arrival * 100))
    station_channels.append(tracce_sac[i].stats['channel'])
    source_magnitude.append(tracce_sac[i].stats['sac']['mag'])
    source_magnitude_type.append("unknown")  # FIXME
    sampling_rate.append(tracce_sac[i].stats['sampling_rate'])
    trace_P_uncertainty_s.append(tracce_sac[i].stats['delta'])
    trace_name.append(str(tracce_sac[i].stats['starttime']) + "." +
                      tracce_sac[i].stats['station'] + "." + tracce_sac[i].stats['channel'])
    
# plt.plot(tracce_sac[0].data)
# plt.axvline(x=int(arrival*100), c="r", ls="--", lw="1")
# plt.show()
#
# np_cosa = np.array(tracce_sac[0].data)
# print(np_cosa, type(np_cosa), type(np_cosa[0]))
# plt.plot(np_cosa)
# plt.axvline(x=int(arrival*100), c="r", ls="--", lw="1")
# plt.show()
# """

