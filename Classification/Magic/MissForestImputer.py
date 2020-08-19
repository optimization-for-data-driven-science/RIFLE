import pandas as pd
import missingpy
from _datetime import datetime
import numpy as np
# mask = np.loadtxt('mask_pattern.csv', delimiter=',')
# data = pd.read_csv('final_data_missing.csv')


# mask = np.loadtxt('mask_pattern_mcar.csv', delimiter=',')
data = pd.read_csv('train_missing_MCAR_100_40_noy.csv')

test = pd.read_csv('test.csv')

test_Y = test[['10']]

test['10'] = np.nan

# test = test[100:200]

total = pd.concat([data, test])
imputer = missingpy.MissForest()


now1 = datetime.now()
current_time = now1.strftime("%H:%M:%S")
print("Starting Time =", current_time)

imputer.fit(data)

total_imputed = imputer.transform(total)


now1 = datetime.now()

current_time = now1.strftime("%H:%M:%S")
print("Final Time =", current_time)

test_imputed = total_imputed[100:]
df = pd.DataFrame(test_imputed)

df.to_csv('MF_MCAR100_40.csv', index=False)
