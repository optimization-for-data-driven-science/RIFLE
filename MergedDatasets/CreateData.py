from math import fabs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy.stats as stat

is_mcar = False
partition_size = 200

data = pd.read_csv("training_data.csv")


data = data.sample(partition_size * 2)

data.to_csv('original_merged2_dataMNAR400.csv', index=None)

data1 = data[:partition_size]
data2 = data[partition_size:2*partition_size]
# data3 = data[2*partition_size:3*partition_size]
# data4 = data[3*partition_size:4*partition_size]
# data5 = data[4*partition_size:5*partition_size]

columns_data1 = []
data_columns = data.columns[51:81]
data1[data_columns] = np.nan

data_columns = data.columns[0:41]
data2[data_columns] = np.nan
# data_columns = data.columns[46:81]
# data2[data_columns] = np.nan

# data_columns = data.columns[:31]
# data3[data_columns] = np.nan
# data_columns = data.columns[61:81]
# data3[data_columns] = np.nan

# data_columns = data.columns[:46]
# data4[data_columns] = np.nan
# data_columns = data.columns[76:81]
# data4[data_columns] = np.nan

# data_columns = data.columns[11:61]
# data5[data_columns] = np.nan


# new_data = pd.concat([data1, data2, data3, data4, data5], axis=0)
new_data = pd.concat([data1, data2], axis=0)
print(new_data)

number_of_data_points = new_data.shape[0]
print(number_of_data_points)

target_column = new_data[['critical_temp']].to_numpy()
# print(target_column)

if is_mcar:

    prob = 0.5
    count = 0
    for i in range(number_of_data_points):
        x = random.random()
        if random.random() < prob:
            target_column[i][0] = np.nan
            count += 1

    new_data[['critical_temp']] = target_column


else:
    count = 0
    from sklearn.preprocessing import StandardScaler
    sc_y = StandardScaler()
    print(target_column[0])
    transformed_target = sc_y.fit_transform(target_column)
    for i in range(len(target_column)):
        if random.random() < stat.norm.cdf(fabs(transformed_target[i][0]) * 1.0 - 1.0):
            target_column[i][0] = np.nan
            count += 1
    new_data[['critical_temp']] = target_column
    print(count)
    print(count / (partition_size*2))
# print(new_data)

mask = new_data.isnull()


# mask = np.ones(mask.shape) - mask.to_numpy()
mask = np.ones(mask.shape) - mask.to_numpy()
print(mask)

fig, ax = plt.subplots()
plt.imshow(mask, cmap='Greys', interpolation='none', aspect='auto')
plt.show()

new_data.to_csv('missing_merged2_dataMNAR400.csv', index=None)
