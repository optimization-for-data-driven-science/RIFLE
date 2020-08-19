import pandas as pd
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

Data1 = pd.read_csv('OriginalSuperConduct.csv')
# Data2 = pd.read_csv('unique_m.csv')

# Data1.drop(['critical_temp'], axis=1, inplace=True)
# Data2.drop(['material'], axis=1, inplace=True)

print(Data1)
# print(Data2)

# final_data = pd.concat([Data1, Data2], axis=1)
final_data = Data1

print(final_data)

# shuffle data
final_data = final_data.sample(frac=1)

number_of_data_points = final_data.shape[0]

train_num = int(number_of_data_points * 0.85)
train_data = final_data[:train_num]
test_data = final_data[train_num:]

print(train_data.shape)
print(test_data.shape)

train_data.to_csv('training_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
