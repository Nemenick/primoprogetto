import geopandas
from Classe_sismogramma_v3 import ClasseDataset
import numpy as np
from matplotlib import pyplot as plt
import folium
#
# hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s.hdf5'
# csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s.csv'
# img_italia = plt.imread('/home/silvia/Documents/GitHub/primoprogetto/img_italia.jpg')
# Data = ClasseDataset()
# Data.leggi_custom_dataset(hdf5in, csvin)
# world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# grafico = world[world.continent == 'Europe'].plot(color='white', edgecolor='black')
# min_lat = np.min(Data.metadata['source_latitude_deg'])
# max_lat = np.max(Data.metadata['source_latitude_deg'])
# min_lon = np.min(Data.metadata['source_longitude_deg'])
# max_lon = np.max(Data.metadata['source_longitude_deg'])
# grafico.set_xlim(min_lon, max_lon)
# grafico.set_ylim(min_lat, max_lat)
#
# hb = grafico.hexbin(x=Data.metadata['source_longitude_deg'],
#                     y=Data.metadata['source_latitude_deg'],
#                     gridsize=200,
#                     cmap='inferno',
#                     bins="log",
#                     zorder=1,
#                     )
# min_lat = np.min(Data.metadata['source_latitude_deg'])
# max_lat = np.max(Data.metadata['source_latitude_deg'])
# min_lon = np.min(Data.metadata['source_longitude_deg'])
# max_lon = np.max(Data.metadata['source_longitude_deg'])
# # grafico.set_xlim(min_lon, max_lon)
# # grafico.set_ylim(min_lat, max_lat)
# grafico.axis([min_lon, max_lon, min_lat, max_lat])
# grafico.set_title("Hexagon binning")
# grafico.set_title("Hexagon binning")
# # cb = figura.colorbar(hb, ax=grafico)
# # cb.set_label('counts')
# # plt.axvline(min_lat, c='navy')
# print(min_lon, max_lon)
# print(min_lat, max_lat)
# # plt.colorbar()
# plt.show()
