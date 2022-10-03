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
viste4 = "/home/silvia/Desktop/viste4"
buone = "/home/silvia/Desktop/viste_buone"

cartelle_path = os.listdir(viste4)
cartelle_path.sort()
for cartella in cartelle_path:
    tracce_sac = read(viste4 + "/" + cartella + "/*Z.sac", format="SAC")
    shutil.move(viste4 + "/" + cartella , buone)
