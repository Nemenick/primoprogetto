a = [i for i in range(10)]
b = [3, 8]

indice = []
for j in range(len(b)):
    for i in range(len(a)):
        print("\n", j, i)
        if b[j] == a[i]:
            indice.append(i)
            break
print(indice)
