import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import math
# from Classe_sismogramma_v3 import ClasseDataset
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def calibration(tentativo: str = '52', nbin = 10):
    percorso = "/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/" + tentativo
    datd = pd.read_csv(percorso + "/Predizioni_test_tentativo_" + tentativo + ".csv",
                       dtype={"y_Mano_test": float, "y_predict_test": float, 'delta_test': float})
    print(datd.keys())
    # traccia,  ymano,  ypredict,   delta
    chiavi  = list(datd.keys())
    predizioni = np.array([[0 for _ in range(3)] for __ in range(nbin)])
    # predizioni [5][0] = pred 5 intervallo
    # predizioni [5][1] = pred 5 intervallo labellate up
    # predizioni [5][2] = pred 5 intervallo labellate down

    for i in range(1, len(datd)):
        indice = math.floor(float(datd[chiavi[2]][i]) * nbin)
        indice = 19 if indice == 20 else indice
        predizioni[indice][0] += 1
        if datd[chiavi[1]][i] > 0.5:
            predizioni[indice][1] += 1
        else:
            predizioni[indice][2] += 1
    predizioni = np.transpose(predizioni)
    print(np.array(predizioni))
    # a = predizioni[2]/predizioni[0]
    # return predizioni[1]/predizioni[0], a[len(a)::-1]
    return predizioni


# TODO reliability
"""
n_bin = 12
ipath = '/home/silvia/Desktop/CFM_images'
x = np.arange(0, n_bin)/n_bin
y_optimum_up = (np.arange(0, n_bin) + 0.5) / n_bin           # TODO caso in cui affidabilità singola classe
# y_optimum_down = y_optimum_up[n_bin::-1]
predizioni = Calibration('52', n_bin)
y1 = predizioni[1] / predizioni[0]
y_excess = np.array([max(0,y1[i]-y_optimum_up[i]) for i in range(len(y1))])
# y2 =  predizioni[2] / predizioni[0]

colors = ['#bdbcdb', '#3a3da7']
cmap = LinearSegmentedColormap.from_list('my_palette', colors)
custom_cmap = [cmap(i/n_bin) for i in range(n_bin)]


fig, ax = plt.subplots()


rects = ax.bar(x, y1-y_excess, width=1/n_bin, label='CFM calibration', edgecolor="black",
               color=custom_cmap, align='edge', zorder=1)
ax.bar(x, y_excess, width=1/n_bin, label='Exceeding calibration', edgecolor="black", color='blue', align='edge',
       zorder=2, hatch='//', bottom=y1-y_excess)
index = 0
for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, max(y1[index],y_optimum_up[index]) , predizioni[1][index],
            ha='center', va='bottom', color='red')
    ax.text(rect.get_x() + rect.get_width() / 2, max(y1[index], y_optimum_up[index]) + 0.05, predizioni[2][index],
            ha='center', va='bottom', color='blue')
    index += 1

# background
ax.bar(x, y_optimum_up, width=1 / n_bin, label='Deficit calibration', edgecolor="red", alpha=0.5, hatch='//', color='red',
       align='edge', zorder=-1)
ax.bar(x, y_optimum_up, width=1 / n_bin, label='Optimal Calibration', align='edge', zorder=2, color=(0, 1, 0, 0.000001),
       edgecolor=(0, 0.65, 0, 1), linewidth=1.5)
ax.grid(color='gray', linestyle=':', linewidth=1, alpha=0.3)
ax.set_axisbelow(True)
# plt.plot([0, 1], [0, 1], linestyle="--", color='grey', linewidth="2")

ax.set_xlim(0, 1.)
ax.set_ylim(0, 1.1)
ax.set_title('P(polarity(x) = up | CFM(x))')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Fraction of positive class (upward)')
plt.legend()
# plt.show()
plt.savefig(ipath + '/Reliability4.jpg')
ECE = sum(predizioni[0]*abs(y1-y_optimum_up))/sum(predizioni[0])
# plt.savefig(ipath + '/Reliability.jpg', dpi=300)
print(ECE)
"""

# TODO grafica mappa
"""
csv_pol = '/home/silvia/Desktop/Pollino_All/Pollino_All_events_lat_lon.csv'
Data_pol = pd.read_csv(csv_pol)
min_lat_pol = 39.4  # np.min(Data.metadata['source_latitude_deg'])
max_lat_pol = 40.5  # np.max(Data.metadata['source_latitude_deg'])
min_lon_pol = 15.5  # np.min(Data.metadata['source_longitude_deg'])
max_lon_pol= 16.6  # np.max(Data.metadata['source_longitude_deg'])
x_pol, y_pol = [min_lon_pol-0.3, min_lon_pol-0.3, max_lon_pol+0.3, max_lon_pol+0.3, min_lon_pol-0.3], \
    [min_lat_pol-0.3, max_lat_pol+0.3, max_lat_pol+0.3, min_lat_pol-0.3, min_lat_pol-0.3]
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
# m_ins.arcgisimage(service='World_Shaded_Relief', xpixels=1500, verbose=True)
m_ins.arcgisimage(xpixels=1500, verbose=True)
m_ins.drawcoastlines(linewidth=0.5)
m_ins.drawparallels(paralleli, labels=[1, 0, 0, 0])
m_ins.drawmeridians(meridiani, labels=[1, 1, 0, 1])
m_ins.drawcountries()
grafico[0].set_title("ciao")
e_t = [43, 45, 9.5, 11.8]
e_v = [37.5, 38.5, 14.5, 16]

x_t = [9.5, 9.5, 11.8, 11.8, 9.5]
y_t = [43, 45, 45, 43, 43]
x_v, y_v = [14.5, 14.5, 16, 16, 14.5], [37.5, 38.5, 38.5, 37.5, 37.5]
grafico[0].plot(x_t, y_t, zorder=2, linewidth=2, color="deeppink")
grafico[0].plot(x_v, y_v, zorder=2, linewidth=2, color="orange")

# plt.hist2d(x=Data['source_longitude_deg'],
#            y=Data['source_latitude_deg'],
#            bins=(208, 200),
#            cmap='inferno',
#            zorder=1,
#            alpha=0.99,
#            norm=colors.LogNorm()
#            )
# plt.colorbar()
m_ins.scatter(x=Data_ins['source_longitude_deg'],
              y=Data_ins['source_latitude_deg'],
              zorder=1,
              s=2,
              marker = "o",
              color="red",
              alpha = 1,
              edgecolor="black",
              linewidth=0.2
              )


m_pol = Basemap(ax = grafico[1],llcrnrlon=min_lon_pol, urcrnrlon=max_lon_pol, llcrnrlat=min_lat_pol,
                urcrnrlat=max_lat_pol, resolution='h')
m_pol.arcgisimage( xpixels=1500, verbose=True)
# m_pol.arcgisimage(service='World_Shaded_Relief', xpixels=1500, verbose=True)
m_pol.drawcoastlines(linewidth=0.5)
m_pol.drawparallels(paralleli_pol, labels=[1, 0, 0, 0])
m_pol.drawmeridians(meridiani_pol, labels=[1, 1, 0, 1])
m_pol.drawcountries()

# plt.hist2d(x=Data['source_longitude_deg'],
#            y=Data['source_latitude_deg'],
#            bins=(208, 200),
#            cmap='inferno',
#            zorder=1,
#            alpha=0.99,
#            norm=colors.LogNorm()
#            )
# plt.colorbar()
m_pol.scatter(x=Data_pol['source_latitude_deg'],
              y=Data_pol['source_longitude_deg'],
              zorder=11,
              s=2,
              marker = "o",
              color="red",
              alpha = 1,
              edgecolor="black",
              linewidth=0.2
              )
inset_ax = inset_axes(grafico[1], width="40%", height="40%", loc=1)

# Plot a third map on the inset axes

min_lon_insert = 8.5
max_lon_insert = 18
min_lat_insert = 37
max_lat_insert = 44

map3 = Basemap(ax=inset_ax, llcrnrlon=min_lon_insert, urcrnrlon=max_lon_insert, llcrnrlat=min_lat_insert,
                urcrnrlat=max_lat_insert, resolution='h')
map3.drawcoastlines(linewidth=0.2)
# map3.arcgisimage( xpixels=1500, verbose=True)
map3.arcgisimage(service='World_Shaded_Relief', xpixels=1500, verbose=True)
map3.drawcoastlines()
map3.drawcountries()
map3.plot(x_pol, y_pol, zorder=2, linewidth=2, color="blue")
print("sciao")
plt.savefig('/home/silvia/Desktop/ITA_pol_scatter.jpg', dpi=500, bbox_inches='tight')
# plt.show()
"""

# TODO grafico di confronto timeshift
"""
pat = 'Predizioni_shift/'
file = [open(pat+"predizioni_shift_Instance_tent_52.txt", "r"), open(pat+"predizioni_shift_Instance_tent_66.txt", "r"),
        open(pat+"predizioni_shift_Instance_tent_79.txt", "r")]
numero_confronto = len(file)
predizioni = [[[] for j in range(3)] for i in range(numero_confronto)]
# file = [open(pat+"predizioni_shift_Instance_tent_52.txt", "r"), open(pat+"predizioni_shift_Instance_tent_53.txt", "r"),
#         open(pat+"predizioni_shift_Instance_tent_66.txt", "r"), open(pat+"predizioni_shift_Instance_tent_79.txt", "r"),
#         open(pat+"predizioni_shift_Instance_tent_39.txt", "r") ]

path_save = '/home/silvia/Desktop'
name_save = "Figure Timeshift"
labels = ["NO timeshift in training set", "Timeshift N=5", "Timeshift N=10"]
# labels = ["Timeshift in training set", "NO Timeshift in training set"]
colori = ["red", "blue", "green","purple"]

for k in range(numero_confronto):
    for line in file[k]:
        a = line.split()
        # print(a)
        predizioni[k][0].append(int(a[0]))
        for i in range(1, 3):
            predizioni[k][i].append(float(a[i]))


print(predizioni)
plt.figure(figsize=(5,6))
for k in range(numero_confronto):
    plt.plot(predizioni[k][0][10:-10], predizioni[k][2][10:-10], label=labels[k], color=colori[k])

plt.title('Test Accuracy')
plt.axhline(0.5, color='k', ls='dashed', lw=1)
plt.axhline(0.75, color='k', ls='dashed', lw=1)
plt.legend()
plt.xlabel('T  (translation samples in test set)')
plt.axvline(linestyle="--", color="red", linewidth="0.5")
plt.savefig(path_save + "/" + name_save, dpi=300)
"""
# plt.savefig('/home/silvia/Desktop/quiii')


# from mpl_toolkits.basemap import Basemap
# import matplotlib.pyplot as plt
# min_lat = 39.5   # np.min(Data.metadata['source_latitude_deg'])
# max_lat = 40.4    # np.max(Data.metadata['source_latitude_deg'])
# min_lon = 15.5    # np.min(Data.metadata['source_longitude_deg'])
# max_lon = 16.6   # np.max(Data.metadata['source_longitude_deg'])
# print(min_lat, max_lat, min_lon, max_lon)
#
# map = Basemap(llcrnrlon=min_lon,  urcrnrlon=max_lon, llcrnrlat=min_lat, urcrnrlat=max_lat, resolution='h')
#
# #http://server.arcgisonline.com/arcgis/rest/services
# # world phisical map
# map.arcgisimage(service='World_Shaded_Relief', xpixels = 1500, verbose= True)




