import datetime

a = datetime.datetime.now()
type(a)
# <class 'datetime.datetime'>
b = datetime.datetime.now()
type(b-a)
# <class 'datetime.timedelta'>
print((b - a).total_seconds())
# 87.51675
b = datetime.datetime(2017, 11, 28, 23, 55, 59, 342380)
a = datetime.datetime(2017, 11, 28, 23, 55, 58, 0)
(b-a).total_seconds()

                  # h   m   s   micro_s
c = datetime.time(0, 0, 59)
d = datetime.time(11, 34, 56, 234566)
# c-d NON E VALIDO!


"""from obspy import UTCDateTime
print(cosini["source_origin_time"][0])
# cos_data = datetime.time(cosini["source_origin_time"][0])
# cos_data
utc_time = UTCDateTime(cosini["source_origin_time"][0])
print(utc_time,type(utc_time))


UTCDateTime(f'{b[0].stats.sac.nzyear}-{str(b[0].stats.sac.nzjday).zfill(3)}T{str(b[0].stats.sac.nzhour).zfill(2)}:{str(b[0].stats.sac.nzmin).zfill(2)}:{str(b[0].stats.sac.nzsec).zfill(2)}.{b[0].stats.sac.nzmsec}')
time_bello = UTCDateTime("2009-001T01:23:34.5")
print(time_bello)

"""
