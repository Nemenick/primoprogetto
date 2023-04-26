import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import math
from Classe_sismogramma_v3 import ClasseDataset
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.lines as mlines

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

def textonly(ax, txt, fontsize = 14, loc = 2, *args, **kwargs):
    at = AnchoredText(txt,
                      prop=dict(size=fontsize),
                      frameon=True,
                      loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    return at

# TODO reliability
# """
n_bin = 12
ipath = '/home/silvia/Desktop/Immagini'
x = np.arange(0, n_bin)/n_bin
y_optimum_up = (np.arange(0, n_bin) + 0.5) / n_bin           # TODO caso in cui affidabilità singola classe
# y_optimum_down = y_optimum_up[n_bin::-1]
predizioni = calibration('52', n_bin)
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
# """

# TODO grafica mappa
"""
csv_pol = '/home/silvia/Desktop/Pollino_All/Pollino_All_events_lat_lon.csv'
Data_pol = pd.read_csv(csv_pol)

# instance rapporto = 1.04 (y/x)
# Centro Pol = 16.1, 39.9

csv_insta = '/home/silvia/Desktop/Instance_Data/Tre_4s/Instance_eventi_lat_lon_dep_rimossi_2.csv'
Ins_scatter = pd.read_csv(csv_insta)
min_lat_ins = 35.1392   # np.min(Data.metadata['source_latitude_deg'])
max_lat_ins = 48.166    # np.max(Data.metadata['source_latitude_deg'])
min_lon_ins = 5.3923    # np.min(Data.metadata['source_longitude_deg'])
max_lon_ins = 18.9612   # np.max(Data.metadata['source_longitude_deg'])
paralleli = np.arange(int(min_lat_ins), max_lat_ins, 3)
meridiani = np.arange(int(min_lon_ins), max_lon_ins, 3)


print(min_lat_ins, max_lat_ins, min_lon_ins, max_lon_ins)

servicee= "NatGeo_World_Map"

fig, grafico = plt.subplots(1, 1, figsize=(6,12))  # figsize=(25, 20)
# m_ins = Basemap(ax = grafico,llcrnrlon=min_lon_ins, urcrnrlon=max_lon_ins, llcrnrlat=min_lat_ins,
#                 urcrnrlat=max_lat_ins, resolution='h')
# m_ins.arcgisimage(service='World_Shaded_Relief', xpixels=1500, verbose=True)
# m_ins.arcgisimage(service=servicee,xpixels=1500, verbose=True)
# m_ins.drawcoastlines(linewidth=0.5)
# m_ins.drawparallels(paralleli, labels=[1, 0, 0, 0])
# m_ins.drawmeridians(meridiani, labels=[1, 1, 0, 1])
# m_ins.drawcountries()
# grafico.set_title("ciao")
e_t = [43, 45, 9.5, 11.8]
e_v = [37.5, 38.5, 14.5, 16]

x_t = [9.5, 9.5, 11.8, 11.8, 9.5]
y_t = [43, 45, 45, 43, 43]
x_v, y_v = [14.5, 14.5, 16, 16, 14.5], [37.5, 38.5, 38.5, 37.5, 37.5]
# grafico.plot(x_t, y_t, zorder=12, linewidth=1.3, color="blue")
# grafico.plot(x_v, y_v, zorder=12, linewidth=1.3, color="orange")
color_Instance = []
col_test="blue"
col_val="orange"
col_train="red"
test_ev , train_ev, val_ev = 0,0,0
for k in range(len(Ins_scatter)):
    if e_t[0] < Ins_scatter["source_latitude_deg"][k] < e_t[1] and e_t[2] \
            < Ins_scatter["source_longitude_deg"][k] < e_t[3]:
        color_Instance.append(col_test)
        test_ev += 1
    else:
        if e_v[0] < Ins_scatter["source_latitude_deg"][k] < e_v[1] and e_v[2] \
             < Ins_scatter["source_longitude_deg"][k] < e_v[3]:
                color_Instance.append(col_val)
                val_ev +=1
        else:
            color_Instance.append(col_train)
            train_ev +=1
print("############################", train_ev,val_ev,test_ev, len(color_Instance))
# plt.hist2d(x=Data['source_longitude_deg'],
#            y=Data['source_latitude_deg'],
#            bins=(208, 200),
#            cmap='inferno',
#            zorder=1,
#            alpha=0.99,
#            norm=colors.LogNorm()
#            )
# plt.colorbar()
# m_ins.scatter(x=Ins_scatter['source_longitude_deg'],
#               y=Ins_scatter['source_latitude_deg'],
#               zorder=11,
#               s=2,
#               marker = "o",
#               color=color_Instance,
#               alpha = 1,
#               edgecolor="black",
#               linewidth=0.2
#               )


# legend_ins = [mlines.Line2D([], [], color=col_train, marker='o', linestyle='None',
#                           markersize=8, label='Training events', markeredgecolor="black"),
#                    mlines.Line2D([], [], color=col_val, marker='o', linestyle='None',
#                           markersize=8, label='Validaton events', markeredgecolor="black"),
#                    mlines.Line2D([], [], color=col_test, marker='o', linestyle='None',
#                           markersize=8, label='Test events', markeredgecolor="black")]


# grafico.legend(handles=legend_ins)
# grafico.set_title("Events in Dataset A")

min_lat_pol = 39.491  # np.min(Data.metadata['source_latitude_deg'])
max_lat_pol = 40.309  # np.max(Data.metadata['source_latitude_deg'])
min_lon_pol = 15.7  # np.min(Data.metadata['source_longitude_deg'])
max_lon_pol= 16.55  # np.max(Data.metadata['source_longitude_deg'])
x_pol, y_pol = [min_lon_pol-0.3, min_lon_pol-0.3, max_lon_pol+0.3, max_lon_pol+0.3, min_lon_pol-0.3], \
    [min_lat_pol-0.3, max_lat_pol+0.3, max_lat_pol+0.3, min_lat_pol-0.3, min_lat_pol-0.3]
paralleli_pol = np.arange(int(min_lat_pol), max_lat_pol, 0.2)
meridiani_pol = np.arange(int(min_lon_pol), max_lon_pol, 0.2)
m_pol = Basemap(ax = grafico,llcrnrlon=min_lon_pol, urcrnrlon=max_lon_pol, llcrnrlat=min_lat_pol,
                urcrnrlat=max_lat_pol, resolution='h')
m_pol.arcgisimage( xpixels=1500, verbose=True)
m_pol.arcgisimage(service=servicee, xpixels=1500, verbose=True)
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
              s=7,
              marker = "o",
              color="deeppink",
              alpha = 1,
              edgecolor="black",
              linewidth=0.5
              )

legend_pol = [mlines.Line2D([], [], color="deeppink", marker='o', linestyle='None',
                          markersize=8, label='Test events', markeredgecolor="black")]


# grafico.legend(handles=legend_pol, loc="lower left")
grafico.set_title("Events in Dataset B")
inset_ax = inset_axes(grafico, width="21.5%", height="31.77%", loc=1, borderpad=0)
min_lon_insert = 13
max_lon_insert = 18
min_lat_insert = 37
max_lat_insert = 44

map3 = Basemap(ax=inset_ax, llcrnrlon=min_lon_insert, urcrnrlon=max_lon_insert, llcrnrlat=min_lat_insert,
                urcrnrlat=max_lat_insert, resolution='h')
map3.drawcoastlines(linewidth=0.2)
map3.arcgisimage(service='World_Shaded_Relief', xpixels=1500, verbose=True)
map3.drawcoastlines()
map3.drawcountries()
map3.plot(x_pol, y_pol, zorder=2, linewidth=2, color="blue")
print("sciao")
plt.savefig('/home/silvia/Desktop/POL_'+servicee+'_ssssssss2catter.jpg', dpi=500, bbox_inches='tight')
plt.show()
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

path_save = '/home/silvia/Desktop/Immagini'
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
plt.legend(fontsize=9.5)
plt.xlabel('T  (translation samples in test set)')
plt.axvline(linestyle="--", color="red", linewidth="0.5")
plt.savefig(path_save + "/" + name_save, dpi=300)
"""

# TODO 4 figure heterogeneous
"""
path_save = "/home/silvia/Desktop/Immagini"
name_save = "figure_heterogeneous"
hdf5u = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/data_U_class34.hdf5'
csvu = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/metadata_U_class34.csv'
hdf5d = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/data_D_class47_54.hdf5'
csvd = '/home/silvia/Desktop/Instance_Data/Tre_4s/Som_updown/secondo_buono/metadata_D_class47_54.csv'
Dati_down = ClasseDataset()
Dati_down.leggi_custom_dataset(hdf5d, csvd)
Dati_up = ClasseDataset()
Dati_up.leggi_custom_dataset(hdf5u, csvu)

# predizioni
pat_pred_up = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/52'+"/secondo_buono_data_U_class34.hdf5tentativo_52.csv"
pat_pred_do = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/52'+"/secondo_buono_data_D_class47_54.hdf5tentativo_52.csv"
pred_up = pd.read_csv(pat_pred_up)
pred_do = pd.read_csv(pat_pred_do)



dati_do = []
metadati_do = []
dati_up = []
metadati_up = []
for i in range(len(Dati_down.sismogramma)):
    dati_do.append(Dati_down.sismogramma[i])
    metadati_do.append([Dati_down.metadata["trace_polarity"][i],
                    Dati_down.metadata["source_magnitude"][i],
                    Dati_down.metadata["trace_Z_snr_db"][i]])

for i in range(len(Dati_up.sismogramma)):
    dati_up.append(Dati_up.sismogramma[i])
    metadati_up.append([Dati_up.metadata["trace_polarity"][i],
                    Dati_up.metadata["source_magnitude"][i],
                    Dati_up.metadata["trace_Z_snr_db"][i]])

print(len(metadati_up), len(metadati_do))
# s = [12,41,10,17] forse scambia 25 41 # I migliori fino ad ora
s = [12,41,10,17] # Quelli buoni (classe 34 up e 47 down)
list_dati = [[np.array(dati_up[s[0]]), np.array(dati_up[s[1]])], [np.array(dati_do[s[2]]), np.array(dati_do[s[3]])]]
list_metadati = [[metadati_up[s[0]], metadati_up[s[1]]], [metadati_do[s[2]], metadati_do[s[3]]]]
list_pred = [[pred_up["y_predict"][s[0]], pred_up["y_predict"][s[1]]], [pred_do["y_predict"][s[2]],pred_do["y_predict"][s[3]]]]

fig, axes = plt.subplots(2,2, figsize=(15.3, 7.5))
for i,j in [[0,0], [0,1], [1,0], [1,1]]:
    a = list_dati[i][j][100:300]
    axes[i][j].plot(list_dati[i][j][100:300]/max(np.max(a),-np.min(a)), color='k')
    axes[i][j].axvline(x=100, c="r", ls="--", lw=1)
    titolino = "M = " + str(list_metadati[i][j][1]) + ", SNR = " + str(round(list_metadati[i][j][2],1)) + "dB"
    axes[i][j].set_title(titolino)
    prob = round(list_pred[i][j]*100,1)
    stringa =  "P$_{assigned}$: " + str(list_metadati[i][j][0]) + "\nP$_{predicted}$: "
    if prob > 50:
        stringa = stringa + "positive [" + str(round(list_pred[i][j]*100,1)) + "%]"
    else:
        stringa = stringa + "negative [" + str(100-round(list_pred[i][j] * 100, 1)) + "%]"
    textonly(axes[i][j], stringa, loc=2, fontsize=12)
    # legend_ins = [mlines.Line2D([], [],   linestyle='None',
    #                            label='Training events'),
    #                    mlines.Line2D([], [],  linestyle='None',
    #                            label='Validaton events')]


    # axes[i][j].legend(handles=legend_ins)
for ax in fig.get_axes():
    ax.label_outer()
fig.supxlabel('Time ($10^{-2} s$)', fontsize=20)
fig.supylabel('Normalized ground motion (counts)', fontsize=20, x=0.05)
plt.savefig(path_save + "/" + name_save, dpi=300)
plt.show()

"""



