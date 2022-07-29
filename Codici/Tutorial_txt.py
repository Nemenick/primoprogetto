#
percorsoclassi = "C:/Users/GioCar/Desktop/Tesi_5/SOM/1classesok4.txt"
# f = open(percorsoclassi, "r")
# classi = f.readlines()
# for i in range(len(classi)):
#    stringa = list(classi[i])
#    stringa = ''.join(stringa[3:len(stringa)-1])
#    print(stringa, type(stringa))
#    classi[i] = int(float(stringa))
#
# print("\n\n",classi)


x = []
with open(percorsoclassi, 'r') as f:
    for line in f:
        if line: #avoid blank lines
            x.append(int(float(line.strip())))
print(x)