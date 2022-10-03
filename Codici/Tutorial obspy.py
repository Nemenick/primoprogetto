import obspy
from obspy import read

# TODO Read single flile
# """
path = "/home/silvia/Desktop/Pollino/*/*Z.sac"
cosa = read(path, format="SAC")
print(cosa, type(cosa))
print("\n", cosa[0], "\n", type(cosa[0]))
print("\ndata", cosa[0].data, "\ntype", type(cosa[0].data))
print("\nstats", cosa[0].stats, "\ntype", type(cosa[0].stats))
print("\nkeys", cosa[0].stats.keys)
print("Ciao", cosa[0].stats['starttime'], str(cosa[0].stats['starttime']))

# """
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
cosa = read("C:/Users/GioCar/Desktop/Tesi_5/20100722022024_M2.0/*Z.sac") # plotta tutte le tracce 
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