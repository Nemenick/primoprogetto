import numpy as np
"""
percorsoclassi = "C:/Users/GioCar/Desktop/Tesi_5/SOM/1classesok4.txt"
x = []
with open(percorsoclassi, 'r') as f:
    for line in f:
        if line: #avoid blank lines
            x.append(int(float(line.strip())))
print(x)
"""

coso = np.array([[3*i,3*i+1,3*i+2] for i in range(4)]+[[10000000000,5555555555,666666666]])
np.savetxt("Testo.txt",coso , fmt='%d')
