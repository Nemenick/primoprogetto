import dask.dataframe as dd
import numpy as np
import time
import pandas as pd

percorsocsv = '/home/silvia/Desktop/Instance_Data/Uno/metadata_Instance_events_selected_Polarity_Velocimeter.csv'
percorsocsvout_pandas = '/home/silvia/Desktop/Instance_Data/Uno/metadata_Instance_events_selected_Polarity_Velocimeter_modificato.csv'
datd = dd.read_csv(percorsocsv, dtype={"trace_P_arrival_sample": int})
metadata = {}
start = time.perf_counter()
for key in datd:
    metadata[key] = np.array(datd[key])
    print("ho caricato la key ", key, time.perf_counter() - start)
dizio = metadata
dizio["centrato"] = False
dizio["demeaned"] = False
datapandas = pd.DataFrame.from_dict(dizio)
datapandas.to_csv(percorsocsvout_pandas, index=False)
