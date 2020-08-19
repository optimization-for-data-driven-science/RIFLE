import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Data normalization:
X = train.drop(['10'], axis=1)
Y = train['10']

sc = StandardScaler()

columns = X.columns
ind = X.index

sc.fit(X)
X = sc.transform(X)
X = pd.DataFrame(X, columns=columns, index=ind)

Y_test = test[['10']]
X_test = test.drop(['10'], axis=1)
X_test = sc.transform(X_test)


clf = LogisticRegression(random_state=0, penalty='none').fit(X, Y)
print(clf.score(X, Y))
print(clf.score(X_test, Y_test))
