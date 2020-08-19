import pandas as pd
import missingpy
from _datetime import datetime

# mask = np.loadtxt('mask_pattern.csv', delimiter=',')
# data = pd.read_csv('final_data_missing.csv')


# mask = np.loadtxt('mask_pattern_mcar.csv', delimiter=',')
data = pd.read_csv('conductivity_MCAR_95_Full_noy.csv')
print(data)
imputer = missingpy.MissForest()

test = pd.read_csv('test_data.csv')

test = test[0:1000]
test['critical_temp'] = 'nan'
print(test)

total = pd.concat([data, test], axis=0)
print(total)

now1 = datetime.now()
current_time = now1.strftime("%H:%M:%S")
print("Starting Time =", current_time)

imputer.fit(data)
print("Done!")
Data_Imputed = imputer.transform(total)


test_imputed = Data_Imputed[18073:]
now1 = datetime.now()
current_time = now1.strftime("%H:%M:%S")
print("Final Time =", current_time)

df = pd.DataFrame(test_imputed)

df.to_csv('totalMF_95_Full_1000.csv', index=False)
