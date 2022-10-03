import obspy
from obspy import read

# Read single flile
"""
cosa = read("C:/Users/GioCar/Desktop/Tesi_5/20100722022024_M2.0/20100722022012.MGR.HHZ.sac")
print(cosa, type(cosa))
print("\n", cosa[0], "\n", type(cosa[0]))
print("\ndata", cosa[0].data, "\ntype", type(cosa[0].data))
print("\nstats", cosa[0].stats, "\ntype", type(cosa[0].stats))
print("\nkeys", cosa[0].stats.keys)"""

# Read folder with wildcards (wildcards sarebbe read *Z.sac)
"""

cosa = read("C:/Users/GioCar/Desktop/Tesi_5/20100722022024_M2.0/*Z.sac")  # legge tutti i sac non zippati
print(cosa, type(cosa))
print("\n", cosa[0], "\n", type(cosa[0]))
print("\ndata", cosa[0].data, "\ntype", type(cosa[0].data))
print("\nstats", cosa[0].stats, "\ntype", type(cosa[0].stats))
print("\nkeys", cosa[0].stats.keys)
"""

# Plot
"""
cosa = read("C:/Users/GioCar/Desktop/Tesi_5/20100722022024_M2.0/*Z.sac") # plotta tutte le tracce 
cosa.plot()
"""

# Tempi arrivo
"""
sia cosa[0]  <class 'obspy.core.trace.Trace'>
inizio = cosa[0].stats['starttime']
print(inizio.datetime, type(inizio.datetime))
"""