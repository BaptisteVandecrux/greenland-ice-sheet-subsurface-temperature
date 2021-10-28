# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
df = pd.read_csv('subsurface_temperature_summary.csv')
df_all= df[['site','latitude','longitude']]
df_gcnet = pd.read_csv('GC-Net_location.csv')[['Name','Northing','Easting']].rename(columns={'Name':'site','Northing':'latitude','Easting':'longitude'})
df_promice = pd.read_csv('PROMICE_coordinates_2015.csv', sep=';')[['Station name', 'Northing', 'Easting']].rename(columns={'Station name':'site','Northing':'latitude','Easting':'longitude'})

df_all = df_all.append(df_gcnet)
df_all = df_all.append(df_promice)
df_all.latitude = round(df_all.latitude,5)
df_all.longitude = round(df_all.longitude,5)
latlon =np.array([str(lat)+str(lon) for lat, lon in zip(df_all.latitude, df_all.longitude)])
uni, ind = np.unique(latlon, return_index = True)
print(latlon[:3])
print(latlon[np.sort(ind)][:3])
df_all = df_all.iloc[np.sort(ind)]
site_uni, ind = np.unique([str(s) for s in df_all.site], return_index = True)
df_all.iloc[np.sort(ind)].to_csv('coords.csv',index=False)