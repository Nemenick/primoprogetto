import numpy as np
from obspy import read, read_inventory
from obspy.io.xseed import Parser


nome = "210920003904.S0114.CHZ"
pat_dataless = "/home/silvia/Desktop/Data/DETECT/transfer_15588_files_ada1a770/dataless/"

#parser = Parser(f"{pat_dataless}{nome[13:18]}.dless") # La funzione Parser ti permette di leggere i dataless  
parser = Parser(f"{pat_dataless}{nome[13:18]}.dataless")

#print(parser)           ## qui scrive a scermo il minimo di informazioni contenute nel dataless
parser.write_xseed(f"/home/silvia/Desktop/{nome}.xml")  # qui scrive in formato station-xml i fattori di conversione
inv = read_inventory (f"/home/silvia/Desktop/{nome}.xml")   #legge lo station xml
st = read(f"/home/silvia/Desktop/Data/DETECT/useful_traces/{nome}.sac") # legge il file da convertire che può essere in mseed oppure anche il formato sac

print(st)
#st.plot
#st.merge()  ## fa il merge di più file
st.detrend()
st.filter(type='highpass', freq=0.5)  ## fa il filtro delle tracce lette
st.remove_response(inventory=inv, output="VEL") # qui viene rimossa in automatico la funzione di risposta dei sensori e i dati vengono convertiti in velocità
                                               #oppure in accelerazione se al posto di VEL scrivi ACC; oppure spostamento se scrivi DISP
st.detrend()
st.filter(type='highpass', freq=5)  ## fa il filtro delle tracce lette

st.write (f'/home/silvia/Desktop/Data/DETECT/{nome}_responseness.sac', format='SAC')     ###qui scrive l'output in velocità in formato mseed o anche in formato SAC se format si scrive SAC

