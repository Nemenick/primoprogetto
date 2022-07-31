import numpy as np
import pandas as pd
"""
a = {}
a["k2"] = []
a["k2"].append(2)

for key in a:
    print(key)

b = {}
for key in a:
    b[key] = []
"""
"""
class mia():

    def istanzia(self):
        self.metadata={"uno": [i for i in range(100)], "due": [2*i for i in range(100)]}

    def elimina(self, vettore_indici):
        for key in self.metadata:
            self.metadata[key] = np.array(np.delete(self.metadata[key], vettore_indici, axis=0))
            self.metadata[key] = list(self.metadata[key])

indici = [2*i for  i in range(40)]
Prova = mia()
Prova.istanzia()
print( Prova.metadata, type(Prova.metadata), type(Prova.metadata["uno"]))
Prova.elimina(indici)
print( Prova.metadata, type(Prova.metadata), type(Prova.metadata["uno"]))"""
