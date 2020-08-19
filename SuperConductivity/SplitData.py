import numpy as np
import pandas as pd

# train_set = pd.read_csv('train.csv')
total_data = pd.read_csv('train.csv')
# uniques = pd.read_csv('unique_m.csv')

# train_set.drop(['critical_temp'], axis=1, inplace=True)
# uniques.drop(['material'], axis=1, inplace=True)


# total_data = pd.concat([train_set, uniques], axis=1)
# total_data = total_data.sample(frac=1).reset_index(drop=True)
# print(total_data.shape)

# total_data['target'] = total_data['critical_temp']
# total_data.drop(['critical_temp'], axis=1, inplace=True)
# print(total_data.shape)
n = total_data.shape[0]

test_data = total_data[0:n//10]
print(test_data.shape)
train_data = total_data[n//10:]
print(train_data.shape)

pd.DataFrame.to_csv(train_data, 'super_conduct_training_data.csv', index=False)
pd.DataFrame.to_csv(test_data, 'super_conduct_test_data.csv', index=False)
