import pandas as pd
import numpy as np
import missingpy


MSK_X = pd.read_csv('Conductor_Mask_X.csv')
MSK_Y = pd.read_csv('Conductor_Mask_Y.csv')


mskX = MSK_X.replace(0, np.nan)
mskY = MSK_Y.replace(0, np.nan)

TrainData = pd.read_csv('training_data.csv')
Y = TrainData[['target']]
print(Y.shape)
X = TrainData.drop(['target'], axis=1)
print(X.shape)

Y = np.multiply(Y, mskY)
X = np.multiply(X, mskX)
Total = pd.concat([X, Y], axis=1)
print(Total.shape)

print(Total)
exit(0)

imputer = missingpy.MissForest()

Data_Imputed = imputer.fit_transform(Total)

df = pd.DataFrame(Data_Imputed)

df.to_csv('MissForestImputed.csv', index=False)
