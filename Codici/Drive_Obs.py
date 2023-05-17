import obspy
import numpy as np
from obspy import read, Stream, Trace
import datetime
import pandas as pd
from matplotlib import pyplot as plt
from Classe_sismogramma_v3 import ClasseDataset

# TODO Np_Data+Dictionary_Metadata from sac
"""
# VEDI PRIMA CODICE IN _Sposta_cartelle

# path = "/home/silvia/Desktop/Pollino/*/*Z.sac"
path = "/home/silvia/Desktop/Waveforms_Pollino_buone/*/*Z.sac"
hdf5out = '/home/silvia/Desktop/Pollino_All_data.hdf5'
csvout = '/home/silvia/Desktop/Pollino_All_metadata.csv'
# path_pandas = "/home/silvia/Desktop/polarity.csv"
Desktop = '/home/silvia/Desktop'
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
source_latitude_deg = []                # ok
source_longitude_deg = []               # ok
source_origin_time = []                 # ok
station_code = []                       # ok
trace_P_arrival_sample = []             # ok
trace_polarity = []                     # ok
trace_P_uncertainty_s = []              # ok
source_magnitude = []                   # ok
source_magnitude_type = []              # ok
sampling_rate = []                      # ok
data = []                               # ok

data_list = ['nzyear', 'nzjday', 'nzhour', 'nzmin', 'nzsec', 'nzmsec']
for i in range(len(tracce_sac)):
    # TODO per arrival sample

    # print(start.datetime, type(start.datetime), type(start))
    d = []
    for j in data_list:
        # print(tracce_sac[i].stats['sac'][j])
        d.append(tracce_sac[i].stats['sac'][j])
    reference = obspy.UTCDateTime(year=d[0], julday=d[1], hour=d[2], minute=d[3], second=d[4], microsecond=d[5]*1000)
    start = tracce_sac[i].stats['starttime']
    # print(reference.datetime, type(reference.datetime), type(reference))
    arrival = (reference.datetime-start.datetime).total_seconds() + tracce_sac[i].stats['sac']['a']
    arr_sample = int(arrival * 100)
    trace_P_arrival_sample.append(arr_sample)

    # TODO per polarity
    pol = tracce_sac[i].stats['sac']['ka']
    if pol.find("U") >= 0:
        trace_polarity.append("positive")
    elif pol.find("D") >= 0:
        trace_polarity.append("negative")
    else:
        trace_polarity.append("undecidable")

    # TODO altri metadata
    station_channels.append(tracce_sac[i].stats['channel'])
    station_code.append(tracce_sac[i].stats['station'])
    source_latitude_deg.append(tracce_sac[i].stats['sac']['evla'])
    if 'o' in tracce_sac[i].stats['sac'].keys():
        source_origin_time.append(reference+tracce_sac[i].stats['sac']['o'])
    else:
        source_origin_time.append(reference)
    source_longitude_deg.append(tracce_sac[i].stats['sac']['evlo'])
    trace_name.append(str(tracce_sac[i].stats['starttime']) + "." +
                      tracce_sac[i].stats['station'] + "." + tracce_sac[i].stats['channel'])
    print("ECCO A VOI LA  I \t\t\t", i, trace_name[i])
    if 'mag' in tracce_sac[i].stats['sac'].keys():
        source_magnitude.append(tracce_sac[i].stats['sac']['mag'])
    else:
        source_magnitude.append("unknown")
    if 'imagtyp' in tracce_sac[i].stats['sac'].keys():
        source_magnitude_type.append("mB")      # FIXME str(tracce_sac[i].stats['sac']['imagtyp'])
    else:
        source_magnitude_type.append("unknown")
    sampling_rate.append(tracce_sac[i].stats['sampling_rate'])
    trace_P_uncertainty_s.append(tracce_sac[i].stats['delta'])

    # TODO np.concatenate (()) DOPPIA PARENTESI!

    data.append(tracce_sac[i].data[arr_sample-200:arr_sample+200])

data = np.array(data)
metadata = {"trace_name": trace_name,
            "source_latitude_deg": source_latitude_deg,
            "source_longitude_deg": source_longitude_deg,
            "source_origin_time": source_origin_time,
            "station_code": station_code,
            "station_channels": station_channels,
            "trace_P_arrival_sample": trace_P_arrival_sample,
            "trace_polarity": trace_polarity,
            "trace_P_uncertainty_s": trace_P_uncertainty_s,
            "source_magnitude": source_magnitude,
            "source_magnitude_type": source_magnitude_type,
            "sampling_rate": sampling_rate
            }
print("\nCIAOOOOOOOOOOOOOOOO", data.shape, len(trace_P_arrival_sample))
print(data[1])

DatasetPollino = ClasseDataset()
DatasetPollino.sismogramma = data
DatasetPollino.metadata = metadata
DatasetPollino.centrato = True
DatasetPollino.demean("rumore")
DatasetPollino.normalizza()
DatasetPollino.crea_custom_dataset(hdf5out, csvout)
# DatasetPollino.plotta(range(len(data)), semiampiezza=130, namepng="Pollino_figure", percosro_cartellla=Desktop)
# plt.plot(tracce_sac[0].data)
# plt.axvline(x=int(arrival*100), c="r", ls="--", lw="1")
# plt.show()
#
# np_cosa = np.array(tracce_sac[0].data)
# print(np_cosa, type(np_cosa), type(np_cosa[0]))
# plt.plot(np_cosa)
# plt.axvline(x=int(arrival*100), c="r", ls="--", lw="1")
# plt.show()
"""

# TODO mseed from Np_Data+Dictionary_Metadata
# """
# in stream.Trace.data metto i data in stream.Trace.stats info standard,
# la domanda è in stream.Trace.stats.mseed posso mettere tutto quello che voglio come dizionario??????????????
hdf5in = '/home/silvia/Desktop/Instance_Data/Tre_4s/data_Velocimeter_4s_Normalizzate_New1-1.hdf5'
csvin = '/home/silvia/Desktop/Instance_Data/Tre_4s/metadata_Velocimeter_4s_Normalizzate_New1-1.csv'

Dati_ori = ClasseDataset()
Dati_ori.leggi_custom_dataset(hdf5in, csvin)


Data_stream = Stream()
# for i in range(len(Dati_ori.sismogramma)):
for i in range(len(10)):
# Create a Trace object with the data. header =... fill my trace.stats
    trace = Trace(data=Dati_ori.sismogramma[i],
              header={'network': 'network_value',
                      'station': 'station_value',
                      'location': 'location_value',
                      'channel': 'channel_value',
                      'starttime': starttime_value,
                      'sampling_rate': sampling_rate_value})
    # trace.stats = metadata
    trace.id = Dati_ori.metadata["trace_name"]  # ok

    Data_stream.append(trace)

# """
