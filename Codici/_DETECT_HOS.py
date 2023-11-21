import _Library_HOS
import numpy as np
from Classe_sismogramma_v3 import ClasseDataset
import pandas as pd
import gc

print("inizio SU DETECT")

# hd = "/home/silvia/Desktop/Data/Pollino_All/Pollino_All_data_extended.hdf5"
# cs = "/home/silvia/Desktop/Data/Pollino_All/Pollino_All_metadata_extended.csv"

hd = "/home/silvia/Desktop/Data/DETECT/Detect_data_picked_extended.hdf5"
cs = "/home/silvia/Desktop/Data/DETECT/Detect_metadata_picked_extended.csv"

D = ClasseDataset()
D.leggi_custom_dataset(hd,cs)
#D.demean()                              # SEVE PERCHE....

uu = pd.DataFrame.from_dict(D.metadata["trace_name"])
uu["trace_P_arrival_sample"] = D.metadata["trace_P_arrival_sample"]

# sampling_rate = 100
# typ_filters = ["lowpass"]
# stats = [_Library_HOS.S_6]
# freq_filters = [1]
# window_widths = [100]
# treshs = [0.25]

#    [     stat           filt   freq wind  th]
# p = [[_Library_HOS.S_6, "highpass", 1, 150, 0.2],
#      [_Library_HOS.S_6, "highpass", 1, 200, 0.1],
#      [_Library_HOS.S_6, "highpass", 2, 150, 0.2],
#      [_Library_HOS.S_6, "highpass", 2, 200, 0.1],
#      [_Library_HOS.S_4, "highpass", 1, 150, 0.2],
#      [_Library_HOS.S_4, "highpass", 1, 200, 0.1],
#      [_Library_HOS.S_4, "highpass", 2, 150, 0.4],
#      [_Library_HOS.S_4, "highpass", 2, 200, 0.4],
#      [_Library_HOS.S_6, "lowpass", 20, 50 , 0.4]]

p = [[_Library_HOS.S_6, "highpass", 1, 150, 0.2],
     [_Library_HOS.S_6, "highpass", 1, 200, 0.1],


     [_Library_HOS.S_4, "highpass", 1, 150, 0.2],
     [_Library_HOS.S_4, "highpass", 1, 200, 0.1],
     

     [_Library_HOS.S_6, "lowpass", 20, 50 , 0.4]]

names = ["S_6","S_6","S_4","S_4","S_6"]
indi = 0
for stat, filt, freq, wind, th in p:
    for ii in range(10):
        gc.collect()
    string = f"stat: {str(names[indi])} type_filter: {filt} filter freq: {freq} window_width: {wind} tresh: {th}"
    print("sto facendo tantecosse", string)
    ons_th = []
    ons_max = []

    
    for i in range(len(D.sismogramma)):                  #TODO FREQ campionamento ATTENTO
        sig = _Library_HOS.freq_filter(D.sismogramma[i], D.metadata["sampling_rate"][i], freq, type_filter= filt)
        onset_th, diff, onset_max,u  = _Library_HOS.get_onset_ori_2(sig, wind, threshold=th, statistics= stat)
        ons_th.append(onset_th)
        ons_max.append(onset_max)
        #onset_1, diff, onset_2  = _Library_HOS.get_onset(sig, window_width, threshold=tresh, statistics= stat)
        # print(onset_1, onset_2)
    ons_th = np.array(ons_th)
    ons_max = np.array(ons_max)
    
    uu = pd.concat([uu,pd.DataFrame.from_dict({f"{string}_ons_th":ons_th})],axis=1)
    uu = pd.concat([uu,pd.DataFrame.from_dict({f"{string}_ons_max":ons_max})],axis=1)
    #uu[f"{string}_ons_th"] = ons_th
    #uu[f"{string}_ons_max"] = ons_max
    uu.to_csv("/home/silvia/Desktop/ONSET_HOS/ONSET_DETECT_alredypicked_get_onset_ori_2_bound1000.csv",index=False)
    indi +=1