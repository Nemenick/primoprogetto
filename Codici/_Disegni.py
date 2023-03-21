import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import math

def Calibration(tentativo: str = '52', nbin = 10):
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


