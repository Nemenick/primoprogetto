import os
from obspy import read
import shutil

"""
Spostacartelle:
Alcune cartelle presentano dei file sac che provando a leggere danno il seguente errore :
obspy.io.sac.util.SacIOError: Actual and theoretical file size are inconsistent.
Actual/Theoretical: 93032/93028
Check that headers are consistent with time series.
Cerco di individuarle e spostare solo quelle buone


"""
all_waveforms = "/home/silvia/Desktop/All_Waveforms_Pollino"
buone = "/home/silvia/Desktop/Waveforms_Pollino_buone"

for cartella in os.listdir(all_waveforms):
    tracce_sac = read(all_waveforms + "/" + cartella + "/*Z.sac", format="SAC")
    shutil.move(all_waveforms + "/" + cartella, buone)







# TODO accedi a lista di sottocartelle NUOVO METODO DI LISTA CARTELLE! (utile se ho anche files tra le cartelle)
"""import os
Xcorso = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/More_1'
for it in os.scandir(Xcorso):
    if it.is_dir():
        print(it.path)"""