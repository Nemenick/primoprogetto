figura, grafico_lon = plt.subplots()
hb = grafico_lon.hexbin(x=Data.metadata['source_longitude_deg'],
                        y=Data.metadata['source_latitude_deg'],
                        gridsize=50,
                        cmap='inferno',
                        bins="log")
min_lat = np.min(Data.metadata['source_latitude_deg'])
max_lat = np.max(Data.metadata['source_latitude_deg'])
min_lon = np.min(Data.metadata['source_longitude_deg'])
max_lon = np.max(Data.metadata['source_longitude_deg'])
grafico_lon.axis([min_lon, max_lon, min_lat, max_lat])
grafico_lon.set_title("Hexagon binning")
grafico_lon.set_title("Hexagon binning")
cb = figura.colorbar(hb, ax=grafico_lon)
cb.set_label('counts')
# plt.axvline(min_lat, c='navy')
print(min_lon, max_lon)
print(min_lat, max_lat)
"""
# plt.colorbar()
"""
plt.show()