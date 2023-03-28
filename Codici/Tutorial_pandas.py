import pandas as pd
csv = '/home/silvia/Desktop/Instance_Data/metadata_Instance_events_v2.csv'
csvout = '/home/silvia/Desktop/Instance_Data/eventi_lat_lon_dep.csv'

dizio = {"ciao": [5,5,5,5,6,7,]}
data = pd.DataFrame.from_dict(dizio)
# Set the index of the DataFrame to the country name
data.drop(data.index[1], inplace=True)
print(False in [i == data.index[i] for i in range(len(data.index))])    # ORA MI DA True!!!


data = data.set_index("station_channels")
data.head()
data = data.drop(["HN", "HL"]) # rimuovo colonne

data = data.set_index("trace_polarity")
data.head()
data = data.drop(["undecidable"])


data.to_csv('/home/silvia/Desktop/ciao.csv')

