import _Library_HOS
import numpy as np
from Classe_sismogramma_v3 import ClasseDataset
import pandas as pd
import gc

print("inizio")

hd = "/home/silvia/Desktop/Data/Pollino_All/Pollino_All_data_extended.hdf5"
cs = "/home/silvia/Desktop/Data/Pollino_All/Pollino_All_metadata_extended.csv"

D = ClasseDataset()
D.leggi_custom_dataset(hd,cs)
D.demean()

uu = pd.DataFrame.from_dict(D.metadata["trace_name"])
uu["trace_P_arrival_sample"] = D.metadata["trace_P_arrival_sample"]

sampling_rate = 100
typ_filter = "lowpass"
stat = _Library_HOS.S_6

# freq_filter = 1
# window_width = 100
# tresh = 0.25

for freq_filter in [8,10,15,20]:
    for window_width in [50,100,150,200]:
        for tresh in [0.2,0.25,0.3,0.4]:
            for ii in range(10):
                gc.collect()
            string = f"filter freq : {freq_filter} window_width: {window_width} tresh: {tresh}"
            print("sto facendo LOWPASS", string)
            ons_th = []
            ons_max = []
            for i in range(len(D.sismogramma)):
                sig = _Library_HOS.freq_filter(D.sismogramma[i], D.metadata["sampling_rate"][i], freq_filter, type_filter= typ_filter)
                onset_th, diff, onset_max,u  = _Library_HOS.get_onset_2(sig, window_width, threshold=tresh, statistics= stat)
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
    uu.to_csv("/home/silvia/Desktop/ONSET_POLLINO_S_6_lowpass.csv",index=False)