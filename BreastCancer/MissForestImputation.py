import pandas as pd
import missingpy
from _datetime import datetime

# mask = np.loadtxt('mask_pattern.csv', delimiter=',')
# data = pd.read_csv('final_data_missing.csv')


# mask = np.loadtxt('mask_pattern_mcar.csv', delimiter=',')
data = pd.read_csv('missing_BC_MCAR30_1.csv')

imputer = missingpy.MissForest()


now1 = datetime.now()
current_time = now1.strftime("%H:%M:%S")
print("Starting Time =", current_time)

Data_Imputed = imputer.fit_transform(data)


now1 = datetime.now()
current_time = now1.strftime("%H:%M:%S")
print("Final Time =", current_time)

df = pd.DataFrame(Data_Imputed)

df.to_csv('MF_BC_MCAR30_1.csv', index=False)
