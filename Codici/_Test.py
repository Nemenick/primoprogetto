import h5py
import math
import seaborn
import dask.dataframe as dd
import matplotlib.pyplot as plt
# from matplotlib import colors
import obspy
import pandas as pd
# import time
# import warnings
import numpy as np
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from Classe_sismogramma_v3 import ClasseDataset
def quadrato(xi, xf, yi, yf):
    plt.axhline(yi,xi,xf)
    plt.axhline(yf, xi, xf)
    plt.axvline(xi, yi, yf)
    plt.axvline(xf, yi, yf)
#
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Create a figure with 2 subplots
fig, axarr = plt.subplots(1, 2, figsize=(12, 6))

# Plot first map on the first subplot
map1 = Basemap(ax=axarr[0], llcrnrlon=-10, llcrnrlat=35, urcrnrlon=30, urcrnrlat=60, resolution='i', epsg=4326)
map1.drawcoastlines()
map1.scatter([5, 10, 20], [40, 50, 55], latlon=True, marker='o', color='red', alpha=0.5)

# Plot second map on the second subplot
map2 = Basemap(ax=axarr[1], llcrnrlon=90, llcrnrlat=-20, urcrnrlon=180, urcrnrlat=30, resolution='i', epsg=4326)
map2.drawcoastlines()
map2.scatter([100, 120, 150], [-10, 0, 20], latlon=True, marker='o', color='blue', alpha=0.5)

# Create an inset axes in the first subplot
inset_ax = inset_axes(axarr[0], width="30%", height="30%", loc=4)

# Plot a third map on the inset axes
map3 = Basemap(ax=inset_ax, llcrnrlon=2, llcrnrlat=42, urcrnrlon=6, urcrnrlat=44, resolution='i', epsg=4326)
map3.drawcoastlines()
map3.scatter([4], [43], latlon=True, marker='o', color='green', alpha=0.5)

# Set titles for the subplots
axarr[0].set_title("Map 1")
axarr[1].set_title("Map 2")
plt.savefig("/home/silvia/Desktop/CIAOUOVO")



"""
x = np.random.normal(size=1000)
y = np.random.normal(size=1000)

hist, xedges, yedges = np.histogram2d(x, y, bins=(50, 50))
print(len(xedges))
size = min((xedges[0]+ xedges[1]),(yedges[0]+yedges[1]))
print(type(hist), hist.shape)
hist[0][0] = 100
hist [0][1]= 50
hist [1][0]= 100
for i in hist:
    for j in i:
        print(j,end='\t')
    print(" ")
print('#################################à#################################à#################################à')
a = hist[len(hist)-1::-1]
for i in a:
    for j in i:
        print(j,end='\t')
    print(" ")
print('#################################à#################################à#################################à')
for i in xedges:
    print(i, end='\t')
print("")
for i in yedges:
    print(i, end='\t')
for j in range(len(yedges)-1):
    for i in range(len(xedges)-1):
        if hist[j][i]>0:  # hist[i][j] > 49
            plt.scatter((xedges[i]+ xedges[i+1])/2,(yedges[j]+yedges[j+1])/2,  marker="s",
                        facecolors=(0, 0.65, 0, 0.00000001), edgecolors=(0, 0.65, 0, 1), zorder=100
                        )
#
#
plt.pcolormesh(xedges, yedges, hist, alpha=0.99, cmap='inferno', norm=colors.LogNorm())
plt.colorbar()
plt.savefig('/home/silvia/Desktop/Pollinoscatter_Tracce_DvavavavPI_2')
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import math
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

csv_pol = '/home/silvia/Desktop/Pollino_All/Pollino_All_events_lat_lon.csv'
Data_pol = pd.read_csv(csv_pol)
min_lat_pol = 39.4  # np.min(Data.metadata['source_latitude_deg'])
max_lat_pol = 40.5  # np.max(Data.metadata['source_latitude_deg'])
min_lon_pol = 15.5  # np.min(Data.metadata['source_longitude_deg'])
max_lon_pol= 16.6  # np.max(Data.metadata['source_longitude_deg'])
paralleli_pol = np.arange(int(min_lat_pol), max_lat_pol, 0.2)
meridiani_pol = np.arange(int(min_lon_pol), max_lon_pol, 0.2)


csv_insta = '/home/silvia/Desktop/Instance_Data/eventi_lat_lon_dep_rimossi_2.csv'
Data_ins = pd.read_csv(csv_insta)
min_lat_ins = 35.1392   # np.min(Data.metadata['source_latitude_deg'])
max_lat_ins = 48.166    # np.max(Data.metadata['source_latitude_deg'])
min_lon_ins = 5.3923    # np.min(Data.metadata['source_longitude_deg'])
max_lon_ins = 18.9612   # np.max(Data.metadata['source_longitude_deg'])
paralleli = np.arange(int(min_lat_ins), max_lat_ins, 3)
meridiani = np.arange(int(min_lon_ins), max_lon_ins, 3)


print(min_lat_ins, max_lat_ins, min_lon_ins, max_lon_ins)


fig, grafico = plt.subplots(1, 2, figsize=(12,12))  # figsize=(25, 20)
m_ins = Basemap(ax = grafico[0],llcrnrlon=min_lon_ins, urcrnrlon=max_lon_ins, llcrnrlat=min_lat_ins,
                urcrnrlat=max_lat_ins, resolution='h')
m_ins.arcgisimage(service='World_Shaded_Relief', xpixels=1500, verbose=True)
m_ins.drawcoastlines(linewidth=0.5)
m_ins.drawparallels(paralleli, labels=[1, 0, 0, 0])
m_ins.drawmeridians(meridiani, labels=[1, 1, 0, 1])
m_ins.drawcountries()
grafico[0].set_title("ciao")
m_ins.scatter(x=Data_ins['source_longitude_deg'],
              y=Data_ins['source_latitude_deg'],
              zorder=1,
              s=2,
              marker = "o",
              color=(0.8, 0, 0),
              alpha = 1,
              edgecolor="black",
              linewidth=0.2
              )


m_pol = Basemap(ax = grafico[1],llcrnrlon=min_lon_pol, urcrnrlon=max_lon_pol, llcrnrlat=min_lat_pol,
                urcrnrlat=max_lat_pol, resolution='h')
m_pol.arcgisimage(service='World_Shaded_Relief', xpixels=1500, verbose=True)
m_pol.drawcoastlines(linewidth=0.5)
m_pol.drawparallels(paralleli_pol, labels=[1, 0, 0, 0])
m_pol.drawmeridians(meridiani_pol, labels=[1, 1, 0, 1])
m_pol.drawcountries()
m_pol.scatter(x=Data_pol['source_latitude_deg'],
              y=Data_pol['source_longitude_deg'],
              zorder=11,
              s=2,
              marker = "o",
              color=(0.8, 0, 0),
              alpha = 1,
              edgecolor="black",
              linewidth=0.2
              )
inset_ax = inset_axes(grafico[1], width="30%", height="30%", loc=0)

# Plot a third map on the inset axes
map3 = Basemap(ax=inset_ax, llcrnrlon=min_lon_ins, urcrnrlon=max_lon_ins, llcrnrlat=min_lat_ins,
                urcrnrlat=max_lat_ins, resolution='c')
map3.drawcoastlines()


print("sciao")
plt.savefig('/home/silvia/Desktop/ITA_pol_scatter.jpg')