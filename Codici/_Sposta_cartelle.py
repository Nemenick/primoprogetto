import os
from obspy import read
import shutil

"""
Alcune cartelle presentano dei file sac che provando a leggere danno il seguente errore :
obspy.io.sac.util.SacIOError: Actual and theoretical file size are inconsistent.
Actual/Theoretical: 93032/93028
Check that headers are consistent with time series.
Cerco di individuarle e spostare solo quelle buone
"""
all_waveforms = "/home/silvia/Desktop/All_Waveforms_Pollino"
buone = "/home/silvia/Desktop/Waveforms_Pollino_buone"

cartelle_path = os.listdir(all_waveforms)
cartelle_path.sort()
for cartella in cartelle_path:
    tracce_sac = read(all_waveforms + "/" + cartella + "/*Z.sac", format="SAC")
    shutil.move(all_waveforms + "/" + cartella, buone)
