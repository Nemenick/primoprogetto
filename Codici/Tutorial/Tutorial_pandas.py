import pandas as pd
import pandas as pd
import openpyxl

# TODO qualcosa
"""

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
"""

# TODO excel sheet
"""


# dataframe Name and Age columns
df = pd.DataFrame({'Name': ['A', 'B', 'C', 'D'],
                   'Age': [10, 0, 30, 50]})
df.append({'Name' : 3, 'Age' : 3},ignore_index=True) # Fixme, devo usare pandas.concat
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('demo.xlsx')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1', index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()
"""

# TODO set index from read
"""
import pandas as pd
si = pd.read_csv("/home/silvia/Desktop/Data/DETECT/Distanze_interstazione_picked_rad.csv").set_index("Unnamed: 0")
si
"""

# TODO remove duplicated columns
"""
data = {'A': [1, 2, 1, 4],
        'B': [5, 6, 7, 8],
        'C': [1, 2, 1, 4]}  # Colonna duplicata
df = pd.DataFrame(data)

df_senza_duplicate = df.T.drop_duplicates(keep='last').T

print(df)
print(df_senza_duplicate)
"""

# TODO insert in pecific location
"""
df.loc[1.5] = ['Jane', 25, 'Madrid']
df = df.sort_index().reset_index(drop=True)

print(df)

# Returns:
#    Name  Age  Location
# 0   Nik   31   Toronto
# 1  Kate   30    London
# 2  Jane   25    Madrid
# 3  Evan   40  Kingston
# 4  Kyra   33  Hamilton

"""