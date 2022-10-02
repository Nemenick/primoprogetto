from datetime import datetime


datetime.now()

type(datetime.now())

a = datetime.now()
b = datetime.now()
type(b-a)
# <class 'datetime.timedelta'>
print((b - a).total_seconds())
# 87.51675
