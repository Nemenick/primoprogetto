from datetime import datetime

a = datetime.now()
type(a)
# <class 'datetime.datetime'>
b = datetime.now()
type(b-a)
# <class 'datetime.timedelta'>
print((b - a).total_seconds())
# 87.51675
