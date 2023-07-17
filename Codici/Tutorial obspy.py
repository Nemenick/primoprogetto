import obspy
from obspy import read
import numpy as np
# Funzione write permette di salvare in sac format.
# purtroppo sembra non possa salvare in unico file ma ogni forma d'onda in un file differente!
# Noto che mseed genera un unico file, molto leggero, vedo se i file mseed e sac condividono le stesse identiche info1
# ok, Mseed va bene!

inutile=False

# TODO Read single flile
"""
# path = "/home/silvia/Desktop/Pollino/*/*Z.sac"
path = '/home/silvia/Desktop/PhaseNet_Prova_1/PhaseNet/test_data/sac/*.sac'
cosa = read(path, format="SAC")
print(cosa, type(cosa))
print("\n", cosa[0], "\n", type(cosa[0]))
print("\ndata", cosa[0].data, "\ntype", type(cosa[0].data))
print("\nstats", cosa[0].stats, "\ntype", type(cosa[0].stats))
print("\nkeys", cosa[0].stats.keys)
print("Ciao", cosa[0].stats['starttime'], str(cosa[0].stats['starttime']))
#cosa.write('/home/silvia/Desktop/PhaseNet_Prova_1/PhaseNet/test_data/sac/unito.mseed')
"""
# TODO Read folder with wildcards (wildcards sarebbe read *Z.sac)
"""

cosa = read("C:/Users/GioCar/Desktop/Tesi_5/20100722022024_M2.0/*Z.sac")  # legge tutti i sac non zippati
print(cosa, type(cosa))
print("\n", cosa[0], "\n", type(cosa[0]))
print("\ndata", cosa[0].data, "\ntype", type(cosa[0].data))
print("\nstats", cosa[0].stats, "\ntype", type(cosa[0].stats))
print("\nkeys", cosa[0].stats.keys)
"""

# TODO Plot
"""

cosa = read("/home/silvia/Desktop/PhaseNet_Prova_1/PhaseNet/test_data/sac/unito.mseed") # plotta tutte le tracce
cosa[0:2].plot()

cosa = read("/home/silvia/Desktop/PhaseNet_Prova_1/PhaseNet/test_data/sac/CI.BOM.2020-10-01T00:00.HHE.sac") # plotta tutte le tracce
cosa.plot()
"""

# TODO Tempi arrivo
"""
sia cosa[0]  <class 'obspy.core.trace.Trace'>
inizio = cosa[0].stats['starttime']
print(inizio.datetime, type(inizio.datetime))
inizio.year
inizio.month
inizio.day


data_list = ['nzyear', 'nzjday', 'nzhour', 'nzmin', 'nzsec', 'nzmsec']
d = []
for i in data_list:
    print(cosa[0].stats['sac'][i])
    d.append(cosa[0].stats['sac'][i])
# Attento in cosa[0].stats['sac']['nzmsec']  è in MILLIsecondi mentre per obspy.UTCDateTime è in MICROsecondi
"""

# TODO confronta Mseed-sac, confronta meta-stats (stats è quello ufficiale, meta sembra non esserci)
"""
# path = "/home/silvia/Desktop/Pollino/*/*Z.sac"
path1 = '/home/silvia/Desktop/PhaseNet_Prova_1/PhaseNet/test_data/sac/*.sac'
cosa1 = read(path1, format="SAC")
path2 = '/home/silvia/Desktop/PhaseNet_Prova_1/PhaseNet/test_data/sac/unito.mseed'
cosa2 = read(path2, format="MSEED")
for tr1,tr2 in zip(cosa1,cosa2):
    cond = True
    for k1,k2 in zip(tr1.meta.keys(), tr2.meta.keys()):
        if k1 == "sac" or k1=="_format": continue
        if k1 != k2:
            print("CHIAVI DIVERSE")

        if tr1.meta[k1] != tr2.meta[k2]:
            cond = False

    for k1,k2 in zip(tr2.meta.keys(), tr2.stats.keys()):
        if k1 == "sac" : continue
        if k1 != k2:
            print("CHIAVI DIVERSE")

        if tr2.meta[k1] != tr2.stats[k2]:
            cond = False

    print(tr1.data.all()==tr2.data.all(), cond)"""

# TODO stream from data
"""# Create an empty Stream object
stream = Stream()

# Create a Trace object with the data
trace = Trace(data=data)

# Assign the metadata dictionary to the stats attribute of the Trace object
trace.stats = metadata

# Add the Trace object to the Stream object
stream.append(trace)
"""

"""

path = '/home/silvia/Desktop/PhaseNet_Prova_1/PhaseNet/test_data/sac/unito.mseed'
cosa = obspy.read(path, format="MSEED")
#print("\nkeys",cosa[0],"\n", cosa[0].stats.keys)

a = obspy.Stream()
a.append(obspy.Trace())
for i in a[0].stats.keys():
    if i == "endtime" or "mseed":
        print(i)
        continue
    print(i)
    a[0].stats[i] = cosa[0].stats[i]
print("\n")
#print(a[0].stats)
a[0].data =np.array([i for i in range(200)])
a.write("ciao.mseed")
b = obspy.read("ciao.mseed")

#for i in a[0].stats.mseed.keys():
 #   print(i)
#print(a[0].stats.keys)

#cosa.write('/home/silvia/Desktop/PhaseNet_Prova_1/PhaseNet/test_data/sac/unito.mseed')
"""



