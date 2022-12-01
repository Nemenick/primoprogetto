import geopandas
from Classe_sismogramma_v3 import ClasseDataset
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as img
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap

file = img.imread('/home/silvia/Desktop/Italia_Tracce_DPI.png')

fig = plt.imshow(file)
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

plt.show()