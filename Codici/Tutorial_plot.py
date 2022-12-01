import geopandas
from Classe_sismogramma_v3 import ClasseDataset
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as img
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap

file = img.imread('/home/silvia/Desktop/Italia_Tracce_DPI.png')

plt.imshow(file)
plt.savefig('/home/silvia/Desktop/Italia_Tracce_DPI_augmented', dpi=500)
plt.show()