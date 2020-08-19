import pandas as pd
import missingpy
from _datetime import datetime

# mask = np.loadtxt('mask_pattern.csv', delimiter=',')
# data = pd.read_csv('final_data_missing.csv')


# mask = np.loadtxt('mask_pattern_mcar.csv', delimiter=',')
data = pd.read_csv('missing_Beta_MCAR_85_10_1000.csv')


nulls = data.isnull().sum()

null_cols = []
columns = []
for i in range(len(nulls)):
    if nulls[i] >= data.shape[0]:
        columns.append(data.columns[i])

data = data.drop(columns, axis=1)

imputer = missingpy.MissForest()


now1 = datetime.now()
current_time = now1.strftime("%H:%M:%S")
print("Starting Time =", current_time)

Data_Imputed = imputer.fit_transform(data)


now1 = datetime.now()
current_time = now1.strftime("%H:%M:%S")
print("Final Time =", current_time)

df = pd.DataFrame(Data_Imputed)

df.to_csv('MF_Beta_MCAR_85_10_1000.csv', index=False)
