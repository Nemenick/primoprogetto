import geopandas
from Classe_sismogramma_v3 import ClasseDataset
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap

hdf5in = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/data_Velocimeter_Buone_4s.hdf5'
csvin = '/home/silvia/Desktop/Instance_Data/Quattro_4s_Buone/metadata_Velocimeter_Buone_4s.csv'
Data = ClasseDataset()
Data.leggi_custom_dataset(hdf5in, csvin)

min_lat = np.min(Data.metadata['source_latitude_deg'])
max_lat = np.max(Data.metadata['source_latitude_deg'])
min_lon = np.min(Data.metadata['source_longitude_deg'])
max_lon = np.max(Data.metadata['source_longitude_deg'])

# TODO basemap
"""
fig, grafico = plt.subplots()
m = Basemap(llcrnrlon=min_lon,  urcrnrlon=max_lon, llcrnrlat=min_lat, urcrnrlat=max_lat, resolution='i')
m.drawcoastlines()
m.fillcontinents()

m.drawparallels(np.arange(36, 52, 2), labels=[1, 1, 0, 1])
m.drawmeridians(np.arange(6, 22, 2), labels=[1, 1, 0, 1])
m.drawcountries()
plt.hist2d(x=Data.metadata['source_longitude_deg'],
           y=Data.metadata['source_latitude_deg'],
           bins=(200, 200),
           cmap='inferno',
           zorder=1,
           alpha=0.99,
           norm=colors.LogNorm()
           )

# hb = grafico.hexbin(x=Data.metadata['source_longitude_deg'],
#                     y=Data.metadata['source_latitude_deg'],
#                     gridsize=200,
#                     cmap='inferno',
#                     bins="log",
#                     zorder=1,
#                     )
# cb = fig.colorbar(hb, ax=grafico)
# cb.set_label('counts')
plt.colorbar()
plt.show()
# plt.savefig('/home/silvia/Desktop/Italia_Bella')
"""

# TODO Geopandas
"""
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
grafico = world[world.continent == 'Europe'].plot(color='white', edgecolor='black', zorder=0)
hb = grafico.hexbin(x=Data.metadata['source_longitude_deg'],
                    y=Data.metadata['source_latitude_deg'],
                    gridsize=200,
                    cmap='inferno',
                    bins="log",
                    zorder=1,
                    )


grafico.set_xlim(min_lon, max_lon)
grafico.set_ylim(min_lat, max_lat)
# grafico.axis([min_lon, max_lon, min_lat, max_lat])
grafico.set_title("Hexagon binning")
grafico.set_title("Hexagon binning")
# cb = figura.colorbar(hb, ax=grafico)
# cb.set_label('counts')
# plt.axvline(min_lat, c='navy')
print(min_lon, max_lon)
print(min_lat, max_lat)
# plt.colorbar()
# plt.savefig('/home/silvia/Desktop/qualcosa.png')
plt.show()
"""

# TODO segmento e quadrato
"""
plt.plot([1,1,2,2,1],[1,2,2,1,1])
plt.show()
"""