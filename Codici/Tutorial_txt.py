
percorsoclassi = "C:/Users/GioCar/Desktop/Tesi_5/SOM/1classesok4.txt"
f = open(percorsoclassi, "r")
classi = f.readlines()
for i in range(len(classi)):
   classi[i]
   stringa = list(classi[i])
   stringa[-1] = "0"
   stringa[-2] = "0"
   stringa = ''.join(stringa)
   print(stringa, type(stringa))
   classi[i] = int(stringa)

print("\n\n",classi)
