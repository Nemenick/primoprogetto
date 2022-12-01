import geopandas
from Classe_sismogramma_v3 import ClasseDataset
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as img
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap

file = img.imread('/home/silvia/Desktop/Rete_2.png')

fig = plt.imshow(file)
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig('/home/silvia/Desktop/Rete_2_augmented.png', dpi=300)
plt.show()