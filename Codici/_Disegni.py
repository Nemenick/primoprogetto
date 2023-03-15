import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


n_bin = 15
x = np.arange(0, n_bin)/n_bin
y = (np.arange(0, n_bin)+0.5)/n_bin
y1 = np.abs(np.random.rand(n_bin)/10 + y - 0.1)

colors = ['#bdbcdb', '#3a3da7']
cmap = LinearSegmentedColormap.from_list('my_palette', colors)
custom_cmap = [cmap(i/n_bin) for i in range(n_bin)]


fig, ax = plt.subplots()


ax.bar(x, y1, width=1/n_bin, label='x2', edgecolor="black", color=custom_cmap, align='edge', zorder=1)
# background
ax.bar(x, y, width=1/n_bin, label='x2', edgecolor="red", alpha=0.5, hatch='//', color='red', align='edge', zorder=-1)
ax.grid(color='gray', linestyle=':', linewidth=1, alpha=0.3)
plt.plot([0, 1], [0, 1], linestyle="--", color='grey', linewidth="2")

ax.set_xlim(0, 1.)
ax.set_title('Histogram-like Graphic')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.legend()
plt.show()
# plt.savefig("C:/Users/GioCar/Desktop/Prova.jpg")

