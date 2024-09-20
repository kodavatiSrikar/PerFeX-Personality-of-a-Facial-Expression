import numpy as np
import pandas as pd

df_au=pd.read_csv('data_range.csv',index_col=False)

df_new=df_au
def spacing_window(data,win_space):
  data=data[data.index%win_space == 0]
  # print(data)
  return data

# Time Shifting
def time_shift(df, shift):
    return df.shift(periods=shift).fillna(method='bfill').fillna(method='ffill')

# Window Slicing
def window_slice(df, window_size=50, step_size=1):
    slices = []
    for start in range(0, len(df) - window_size + 1, step_size):
        slices.append(df.iloc[start:start + window_size].reset_index(drop=True))
    return slices

df_au=pd.read_csv('data_range.csv')
id_count = max(df_au['ID'])
x = max(df_au['ID'])
y = min(df_au['ID'])


for i in range(y, x+1):

	time_series_df = df_au[df_au['ID'] == i]
	# time_series_df= df_au[df_au['File']=='XQ3do4LDFv8.000.mp4']

	# Example Usage
	print(id_count)
	print(len(df_new))
	print(len(time_series_df))
	warped_df = spacing_window(time_series_df,2)
	# print(len(warped_df))
	id_count=id_count+1
	warped_df['ID']=id_count
	print(warped_df.head())
	print(df_new.head())
	df_new=pd.concat([df_new, warped_df])
	shifted_df = time_shift(time_series_df, 5)
	# print(len(shifted_df))
	id_count=id_count+1
	shifted_df['ID']=id_count
	df_new=pd.concat([df_new, shifted_df])
	sliced_dfs = window_slice(time_series_df,100,100)
	for i in sliced_dfs:
		# print(i.head(10))
		id_count=id_count+1
		i['ID']=id_count
		df_new=pd.concat([df_new, i])
		# print(len(i))


df_new.to_csv('data_augumented.csv',index=False) 

