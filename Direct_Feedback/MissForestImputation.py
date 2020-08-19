import pandas as pd
import missingpy
from _datetime import datetime

data = pd.read_csv('feedback_missing_MNAR_P60_5000.csv')

imputer = missingpy.MissForest()


now1 = datetime.now()
current_time = now1.strftime("%H:%M:%S")
print("Starting Time =", current_time)

Data_Imputed = imputer.fit_transform(data)


now1 = datetime.now()
current_time = now1.strftime("%H:%M:%S")
print("Final Time =", current_time)

df = pd.DataFrame(Data_Imputed)

df.to_csv('MF_feedback_MNAR_P60_5000.csv', index=False)
