import obspy
import pandas as pd
from obspy import read, Trace, Stream, UTCDateTime
from Classe_sismogramma_v3 import ClasseDataset
import numpy as np


def instanceType_to_Mseed(data : ClasseDataset, dataset_ori: str, file: str ):
    p_arrival_time = 2
    if dataset_ori.lower() != "instance" and dataset_ori.lower() != "ross" and dataset_ori.lower() != "scsn":
        print("ATTENTO HAI SBAGIATO METODO !, non so cosa fare")
        return
    data_st = Stream()
    # per il pandas dataframe, uso .iloc Ho bisogno sia indicizzato BENE
    
    if dataset_ori.lower() == "instance":
        print("STO USANDO COME INSTANCE")
        for i in range(len(data.sismogramma)):
            data_trace = np.array(data.sismogramma[i], dtype="float32")
            meta_trace = data.metadata.iloc[i]
            nome_split = meta_trace["trace_name"].split(".")

            trace_obs = Trace(data=data_trace,
                        header={'network': nome_split[1],
                                'station': nome_split[2],
                                'channel': nome_split[4],
                                'starttime': UTCDateTime(meta_trace["trace_start_time"] if "trace_start_time" in meta_trace.keys() else 0),
                                
                                'sampling_rate': 100.0})
            trace_obs.id = f"{nome_split[0]}.{nome_split[1]}.{nome_split[2]}.{nome_split[4]}"
            data_st.append(trace_obs)


    if dataset_ori.lower() == "ross" or dataset_ori.lower() == "scsn":
        print("STO USANDO COME ROSS")
        for i in range(len(data.sismogramma)):
            data_trace = np.array(data.sismogramma[i], dtype="float32")
            meta_trace = data.metadata.iloc[i]
            nome_split = meta_trace["trace_name"].split(".")

            trace_obs = Trace(data=data_trace,
                        header={'network': nome_split[0][2:],
                                'station': nome_split[1],
                                'channel': nome_split[2][:-1],
                                'starttime': UTCDateTime(meta_trace["trace_start_time"] if "trace_start_time" in meta_trace.keys() else 0),
                                
                                'sampling_rate': 100.0})
            data_st.append(trace_obs)
    data_st.write(file, format="SAC")