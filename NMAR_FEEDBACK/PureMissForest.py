import pandas as pd
import missingpy
from _datetime import datetime
import numpy as np

data = pd.read_csv('feedback_data_missingNMAR50n1K.csv')

print(data)

imputer = missingpy.MissForest()


now1 = datetime.now()
current_time = now1.strftime("%H:%M:%S")
print("Starting Time =", current_time)

imputer.fit(data)
Data_Imputed = imputer.transform(data)


now1 = datetime.now()
current_time = now1.strftime("%H:%M:%S")
print("Final Time =", current_time)

df = pd.DataFrame(Data_Imputed)

test = pd.read_csv('test_data.csv')

Y_test = test[['280']].copy()

test[['280']] = np.nan

new_data = pd.concat([test, data], axis=0)

test_number = test.shape[0]

test_imputed = imputer.transform(new_data)
test_df = pd.DataFrame(test_imputed)

test_df = test_df[0:test_number]

Y_hat = test_df[[test_df.columns[-1]]]


mse = np.linalg.norm(Y_test - Y_hat) ** 2 / Y_test.shape[0]

print("Pure MissForest:")
print("Test MSE: ", mse)
print("Test RMSE: ", np.sqrt(mse))
print(np.sqrt(Y_test.var()))
exit(0)
