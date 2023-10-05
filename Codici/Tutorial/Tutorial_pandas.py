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
data = data.drop(["HN", "HL"]) # rimuovo dati che hanno questo valore di colonne

data = data.set_index("trace_polarity")
data.head()
data = data.drop(["undecidable"])

# Delete Rows by Checking Conditions
df = pd.DataFrame(technologies)
df1 = df.loc[df["Discount"] >=1500 ]
print(df1)

data.to_csv('/home/silvia/Desktop/ciao.csv')

pred.iloc[:,0:4]                # considera tutte le righe delle colonne 0:4
pred[pred["predizione_0"]<0.5]  # seleziona con singola condizione
ani[(ani > 0.3) & (ani < 0.7)]  # selezione condizioni multiple
