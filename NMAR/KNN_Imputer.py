import pandas as pd
import numpy as np
import missingpy
from datetime import datetime


mask = np.loadtxt('mask_pattern.csv', delimiter=',')

data = pd.read_csv('final_data_missing.csv')

print(data)

imputer = missingpy.KNNImputer()


now1 = datetime.now()
current_time = now1.strftime("%H:%M:%S")
print("Current Time =", current_time)

Data_Imputed = imputer.fit_transform(data)


now2 = datetime.now()
current_time = now2.strftime("%H:%M:%S")
print("Current Time =", current_time)

df = pd.DataFrame(Data_Imputed)

df.to_csv('KNNImputed.csv', index=False)
