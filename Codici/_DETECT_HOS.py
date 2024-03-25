import _Library_HOS
import numpy as np
from Classe_sismogramma_v3 import ClasseDataset
import pandas as pd
import gc
from obspy import UTCDateTime

print("inizio SU DETECT inizio")

# hd = "/home/silvia/Desktop/Data/Pollino_All/Pollino_All_data_extended.hdf5"
# cs = "/home/silvia/Desktop/Data/Pollino_All/Pollino_All_metadata_extended.csv"

# hd = "/home/silvia/Desktop/Data/DETECT/Detect_data_picked_extended.hdf5"
# cs = "/home/silvia/Desktop/Data/DETECT/Detect_metadata_picked_extended.csv"

hd = "/home/silvia/Desktop/Data/DETECT/Detect_all_data_extended.hdf5"
cs = "/home/silvia/Desktop/Data/DETECT/Detect_all_metadata_extended.csv"

D = ClasseDataset()
D.leggi_custom_dataset(hd,cs)                                       # Dataset containing the traces (D.seismogram) and other metadata (D.metadata)


uu = pd.DataFrame.from_dict(D.metadata["trace_name"])               # used to generate the output in a CSV file, save the onset for each trace 
uu["trace_P_arrival_sample"] = D.metadata["trace_P_arrival_sample"]


p = [[_Library_HOS.S_4, "bandpass", [1,30], 300, [0.1,0.2,0.3,0.4]]]
post_origin = 10

names = ["S_4","S_6","S_6","S_6","S_6","S_6","S_6","S_6","S_6",
         "S_4","S_4","S_4","S_4","S_4","S_4","S_4","S_4","S_4"]         # used to generate a string that report the used statistic
indi = 0
for stat, filt, freq, wind, th in p:                                    # cycle on settings
    for ii in range(10):
        gc.collect()    # free some memory, if needed

    string = f"stat: {str(names[indi])} type_filter: {filt} filter freq: {freq} window_width: {wind} tresh:"   # diventerà la key del 
    print("sto facendo: ",hd,"\n", string)
    ons_th = [[] for i in range(len(th)) ]                              # I use different threshold for same setting
    ons_max = []
    
    for i in range(len(D.sismogramma[0:30000])): 

        # register the sample corresponding to the event origin
        or_s =  int((UTCDateTime(D.metadata["source_origin_time"][i])- UTCDateTime(D.metadata["trace_start_time"][i]))*D.metadata["sampling_rate"][i])

        inizio = or_s-wind if or_s-wind >0  else 0
                                        # I extract the portion of the waveform from the arrival until post_origin seconds after
        sig = _Library_HOS.freq_filter(D.sismogramma[i][inizio:or_s+post_origin*int(D.metadata["sampling_rate"][i])], D.metadata["sampling_rate"][i], freq, type_filter= filt)
        onset_th, diff, onset_max,u,hoss  = _Library_HOS.get_onset_4(sig, wind, threshold=th, statistics= stat) # get onsets
        for j in range(len(th)):
            ons_th[j].append(onset_th[j] + or_s-wind)
        ons_max.append(onset_max + or_s-wind)
        
        # "Test che succede su noise!"
        """try:
            inizio = or_s-wind- 4*int(D.metadata["sampling_rate"][i])if or_s-wind >0  else 0
            sig = _Library_HOS.freq_filter(D.sismogramma[i][inizio:inizio+4*int(D.metadata["sampling_rate"][i])], D.metadata["sampling_rate"][i], freq, type_filter= filt)
            onset_th, diff, onset_max,u  = _Library_HOS.get_onset_4(sig, wind, threshold=th, statistics= stat) # get onsets
            
            for j in range(len(th)):
                ons_th[j].append(onset_th[j] + or_s-wind)
            ons_max.append(onset_max + or_s-wind)
        except:
            for j in range(len(th)):
                ons_th[j].append(1000000000)
            ons_max.append(1000000000)"""

    for j in range(len(th)):
        ons_th_tmp = np.array(ons_th[j])
        uu = pd.concat([uu,pd.DataFrame.from_dict({f"{string}_ons_th={th[j]}":ons_th_tmp})],axis=1)

    ons_max = np.array(ons_max)
    uu = pd.concat([uu,pd.DataFrame.from_dict({f"{string}_ons_max":ons_max})],axis=1)
    uu.to_csv("/home/silvia/Desktop/ONSET_HOS/ONSET_DETECT_whole_10s_dopo_inizio_primi_30000.csv",index=False)
    indi +=1