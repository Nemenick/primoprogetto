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
import seaborn
import plotly.graph_objects as go

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

# TODO reliability - ECE
"""
n_bin = 10
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

fig, ax = plt.subplots(figsize=(9,7))

rects = ax.bar(x, y1-y_excess, width=1/n_bin, label='', edgecolor="black",
               color=custom_cmap, align='edge', zorder=1)
ax.bar(x, y_excess, width=1/n_bin, label='Exceeding calibration', edgecolor="black", color='blue', align='edge',
       zorder=2, hatch='//', bottom=y1-y_excess)
index = 0
for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, max(y1[index],y_optimum_up[index]) , predizioni[1][index],
            ha='center', va='bottom', color='red', fontsize=14)
    ax.text(rect.get_x() + rect.get_width() / 2, max(y1[index], y_optimum_up[index]) + 0.05, predizioni[2][index],
            ha='center', va='bottom', color='blue', fontsize=14)
    index += 1

# background
ax.bar(x, y_optimum_up, width=1 / n_bin, label='Deficit calibration', edgecolor="red", alpha=0.5, hatch='//', color='red',
       align='edge', zorder=-1)
ax.bar(x, y_optimum_up, width=1 / n_bin, label='Optimal Calibration', align='edge', zorder=2, color=(0, 1, 0, 0.000001),
       edgecolor=(0, 0.65, 0, 1), linewidth=1.5)
ax.grid(color='gray', linestyle=':', linewidth=2, alpha=0.3)
ax.set_axisbelow(True)
# plt.plot([0, 1], [0, 1], linestyle="--", color='grey', linewidth="2")

ax.set_xlim(0, 1.)
ax.set_ylim(0, 1.1)
ax.set_title('CFM calibration', fontsize=18.5)
ax.set_xlabel('Predicted probability', fontsize=17.5)
ax.set_ylabel('Fraction of positive class (upward)', fontsize=17.5)
ax.tick_params(axis='both', which='major', labelsize=13.5)
plt.legend(fontsize=15.5)

ECE = sum(predizioni[0]*abs(y1-y_optimum_up))/sum(predizioni[0])
print(ECE)
plt.savefig(ipath + '/Reliability11.jpg', dpi=300, bbox_inches='tight')
plt.show()
"""

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
plt.figure(figsize=(3.34646,4.01575))
for k in range(numero_confronto):
    plt.plot(predizioni[k][0][10:-10], predizioni[k][2][10:-10], label=labels[k], color=colori[k])

plt.title('Test Accuracy')
plt.axhline(0.5, color='k', ls='dashed', lw=1)
plt.axhline(0.75, color='k', ls='dashed', lw=1)
plt.legend(fontsize=9.5)
plt.xlabel('T  (translation samples in test set)')
plt.ylabel('Accuracy')
plt.axvline(linestyle="--", color="red", linewidth="0.5")
plt.savefig(path_save + "/" + name_save + ".jpg", dpi=300, bbox_inches='tight')
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
    textonly(axes[i][j], stringa, loc=2, fontsize=13)
    # legend_ins = [mlines.Line2D([], [],   linestyle='None',
    #                            label='Training events'),
    #                    mlines.Line2D([], [],  linestyle='None',
    #                            label='Validaton events')]


    # axes[i][j].legend(handles=legend_ins)
for ax in fig.get_axes():
    ax.label_outer()
fig.supxlabel('Time ($10^{-2} s$)', fontsize=20)
fig.supylabel('Normalized ground motion', fontsize=20, x=0.05)
plt.savefig(path_save + "/" + name_save, dpi=300, bbox_inches='tight')
plt.show()

"""

# TODO matrice confusione Predizione TEST
"""
# "y_Mano_test", "y_predict_test"
fig_name = ['Ross_Shift_tentativo_52']
titoli = ["SCSN test set"]                                  # TODO cambia
path = '/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi'
ipath = '/home/silvia/Desktop/'
predizione = '/Predizioni_Ross_shift+1_tentativo_'  # '/Predizioni_test_tentativo_'
tentativi = ['52']

le = len(tentativi)

# predizioni["y_Mano_test"][j] == 1 and predizioni["delta_test"][j] < 0.5:
# predizioni["y_Mano_pol"][j] == 1 and predizioni["delta_val"][j] < 0.5:
for i in range(le):
    tp, tn, fp, fn = 0, 0, 0, 0
    # predizioni = pd.read_csv(path + '/' + tentativi[i] + predizione + tentativi[i] + '.csv') # TODO
    predizioni = pd.read_csv(path + '/' + tentativi[i] + predizione + tentativi[i] + '.csv')
    for j in range(len(predizioni["traccia"])):
        # predizioni["y_Mano_test"][j] == 1 and predizioni["delta_test"][j] < 0.5:          # TODO
        # predizioni["y_Mano_pol"][j] == 1 and predizioni["delta_val"][j] < 0.5:

        if predizioni["y_Mano"][j] == 1 and predizioni["delta"][j] < 0.5:
            tp += 1
        if predizioni["y_Mano"][j] == 0 and predizioni["delta"][j] < 0.5:
            tn += 1
        if predizioni["y_Mano"][j] == 1 and predizioni["delta"][j] >= 0.5:
            fn += 1
        if predizioni["y_Mano"][j] == 0 and predizioni["delta"][j] >= 0.5:
            fp += 1
    print("SONO TUTTI ? (tutti), tp+tn+..", len(predizioni["y_predict"]), tp+tn+fp+fn)
    print(tp, fn, "\n", fp, tn)
    print((tp+tn) / (fp+fn+tn+tp))
    df = pd.DataFrame([[tp, fn], [fp, tn]], columns=["Positive", "Negative"])

    # plot a heatmap with annotation
    ax = seaborn.heatmap(df, annot=True, fmt=".7g", annot_kws={"size": 20}, cmap="Blues", cbar=False)
    plt.xlabel("Predicted polarity (network)", fontsize=23, labelpad=33)
    plt.ylabel("Assigned polarity (catalogue)", fontsize=23, labelpad=33)
    plt.title(titoli[i], fontsize=23, pad=30)
    ax.set_yticklabels(['Positive', 'Negative'], fontsize=15)
    ax.set_xticklabels(['Positive', 'Negative'], fontsize=15)
    ax.text(-0.1, 1.35,  r'$\mathbf{B}$', transform=ax.transAxes, fontsize=30,
            verticalalignment='top')
    plt.savefig(ipath+'/Confusion_matrix_'+fig_name[i]+".jpg", bbox_inches='tight', dpi=300)
    plt.show()
"""

# TODO non so cosa sia
"""
path_save = "/home/silvia/Desktop"
name_save = "Figure_Hara_Misclassified.jpg"
hH = "/home/silvia/Desktop/Hara/Test/Hara_test_data.hdf5"
cH = "/home/silvia/Desktop/Hara/Test/Hara_test_metadata.csv"
DataH = ClasseDataset()
DataH.leggi_custom_dataset(hH, cH)

indi = [1416, 39, 2397, 139]
# indi = [139, 180, 265, 439]
titoli = [["A", "B"], ["C", "D"]]

# predizioni

pat_pred = "/home/silvia/Documents/GitHub/primoprogetto/Codici/Tentativi/55/Predizioni_Hara_tentativo_55.csv"
pred = pd.read_csv(pat_pred)


fig, axes = plt.subplots(2,2, figsize=(15.3, 7.5))
for i,j in [[0,0], [0,1], [1,0], [1,1]]:
    a = DataH.sismogramma[indi[2*i+j]]
    axes[i][j].plot(a/max(np.max(a),-np.min(a)), color='k')
    axes[i][j].axvline(x=75, c="r", ls="--", lw=0.5)
    titolino = titoli[i][j]
    axes[i][j].set_title(titolino, fontweight="bold")
    
    prob = round(pred["y_predict"][indi[2*i+j]]*100,1)
    stringa =  "P$_{assigned}$: " + str(DataH.metadata["trace_polarity"][indi[2*i+j]]) + "\nP$_{predicted}$: "
    if prob > 50:
        stringa = stringa + "positive [" + str(prob) + "%]"
    else:
        stringa = stringa + "negative [" + str(100-prob) + "%]"

    textonly(axes[i][j], stringa, loc=2, fontsize=13)
    # legend_ins = [mlines.Line2D([], [],   linestyle='None',
    #                            label='Training events'),
    #                    mlines.Line2D([], [],  linestyle='None',
    #                            label='Validaton events')]


    # axes[i][j].legend(handles=legend_ins)
for ax in fig.get_axes():
    ax.label_outer()
fig.supxlabel('Time ($10^{-2} s$)', fontsize=20)
fig.supylabel('Normalized ground motion', fontsize=20, x=0.05)
plt.savefig(path_save + "/" + name_save, dpi=300,bbox_inches='tight')
plt.show()
"""

# TODO  DETECT data : plot eventi picked vs size vs magnitudo (colorscale)
"""
# plot picked in base a magnitudo e num. pick
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
import math

map_events_picked = pd.read_csv("/home/silvia/Desktop/Data/DETECT/Map_events_Picks.csv")
colorscale_custom = [
    [0.0, 'red'],           # a 0 rossi
    [0.042, 'red'],         # tra 1 e 6 rossi
    [0.069, 'blue'],        # Valori tra 7 e 10 saranno rosso-blu
    [0.14, 'green'],        # Valori tra 10 e 20 saranno blu-verdi
    [1.0, 'green'],         # Valori superiori a 20 saranno verdi
]

sizes = []
for i in map_events_picked["source_magnitude"]:
    if not math.isnan(i):
        sizes.append(i*2.0+4.5)
    else:
        sizes.append(1.5)

df = map_events_picked[["source_latitude_deg","source_longitude_deg", "source_depth"]]


# Create an interactive 3D scatter plot
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=df['source_latitude_deg'],
    y=df['source_longitude_deg'],
    z=df['source_depth'],
    mode='markers',
    marker=dict(
        size=sizes,
        opacity=0.7,
        color=map_events_picked["number_P_picks"],
        colorscale=colorscale_custom,  # Usa la colormap personalizzata
        colorbar=dict(title='Color Scale', tickvals=[7, 10, 20], ticktext=['7<: Red','10 Blue', '>20: green']),  # Aggiungi la colorbar personalizzata
    ),
    name='Picked'
))

fig.update_layout(
    scene=dict(
        xaxis_title='Latitude',
        yaxis_title='Longitude',
        zaxis_title='Depth'
    ),
    title='Mappa eventi in base a numero di tracce con pick'
)

# Save the plot to an HTML file
fig.write_html("/home/silvia/Desktop/Plot_qualcosa.html")
"""