import math
from math import sqrt
import pandas as pd
import numpy as np
import glob
import missingpy
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = pd.read_csv('StandardBlogDataMissing.csv')
Y = pd.read_csv('StandardBlogDataMissingLabel.csv')

Y.rename(columns={'0': 'Y'}, inplace=True)
Y.loc[Y['Y'] == -1.0] = np.nan
Total = pd.concat([X, Y], axis=1)

# imputer = missingpy.MissForest()

imputer = missingpy.MissForest()

Data_Imputed = imputer.fit_transform(Total)

df = pd.DataFrame(Data_Imputed)

# df.to_csv('ImputedWithMissForest.csv')
df.to_csv('ImputedWithMF.csv')
