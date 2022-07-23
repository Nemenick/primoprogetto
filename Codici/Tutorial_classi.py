
"""class Classe1:


    def __init__(self):
        self.centrato = False
        a=3
    def fun1(self):
        a = 3
        self.b=0
    def fun2(self):
        print(self.centrato)

d=Classe1()
d.fun2()
d.fun1()
print(d.b)
print(d.a)"""
"""class Classe2:

    def fun1(self, errore=True):
        if errore:
            return 1
        else:
            return 0

    def fun2(self, **kargs):
        print("questo è kargs", kargs)
        esito=self.fun1(kargs)
        esito = self.fun1(kargs["errore"])   # PERCHE' VA COSI'?
        print(esito)

d = Classe2()
d.fun2(errore=False)"""
