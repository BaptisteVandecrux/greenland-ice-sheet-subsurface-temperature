import pandas as pd

df = pd.read_excel('firn_temperature_summary.xlsx')

df['date'] =df['date'].astype(str)
df.loc[df['date'].str.len()==4, 'date'] = [x + '-07-01' for x in df.date if len(x)==4]
df['date'] = pd.to_datetime(df['date'])

for site in df.site.unique():
    if len(df.loc[df.site==site])>10:
        print(site)
        # input()
        tmp_mean = df.loc[df.site==site].set_index('date').resample('M').mean().reset_index()
        tmp = df.loc[df.site==site].set_index('date').resample('M').first().reset_index()
        tmp['temperatureObserved'] = tmp_mean['temperatureObserved']
        df = df[df.site != site]
        df = df.append(tmp)
        
df.to_csv('firn_temperature_summary_monthly.csv')
