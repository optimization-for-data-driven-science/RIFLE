import numpy as np
import pandas as pd

data = pd.read_csv("YearPredictionMSD.csv", header=None)

data1 = data[:1000]
data2 = data[1000:2000]
data3 = data[2000:3000]
data4 = data[3000:4000]

columns_data1 = []
data_columns = data.columns[31:]
data1[data_columns] = np.nan

data_columns = data.columns[1:20]
data2[data_columns] = np.nan
data_columns = data.columns[51:]
data2[data_columns] = np.nan

data_columns = data.columns[1:40]
data3[data_columns] = np.nan
data_columns = data.columns[71:]
data3[data_columns] = np.nan

data_columns = data.columns[6:32]
data4[data_columns] = np.nan
data_columns = data.columns[38:70]
data4[data_columns] = np.nan


new_data = pd.concat([data1, data2, data3, data4], axis=0)
print(new_data)
