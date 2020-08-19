import pandas as pd
import missingpy
from _datetime import datetime
import numpy as np

# X = np.loadtxt('MCARBeta__P80_X2000.csv', delimiter=',')
# Y = np.loadtxt('MCARBeta__P80_Y2000.csv', delimiter=',')
# Y = Y[:, np.newaxis]
# data = np.concatenate((X, Y), axis=1)

data = pd.read_csv('sparse_data_mcar40.csv')

"""
df = pd.DataFrame(data)
df.to_csv('Synthetic_P80_2K.csv', index=False)
exit(0)
"""
imputer = missingpy.MissForest()


now1 = datetime.now()
current_time = now1.strftime("%H:%M:%S")
print("Starting Time =", current_time)

Data_Imputed = imputer.fit_transform(data)


now1 = datetime.now()
current_time = now1.strftime("%H:%M:%S")
print("Final Time =", current_time)

df = pd.DataFrame(Data_Imputed)

df.to_csv('mf_sparse_data_mcar40.csv', index=False)
