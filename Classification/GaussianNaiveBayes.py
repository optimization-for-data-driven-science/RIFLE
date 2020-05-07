import pandas as pd
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

Y = train['Y']
X = train.drop(['Y'], axis=1)

Y_test = test['Y']
X_test = test.drop(['Y'], axis=1)

print(X.shape)
print(Y.shape)

clf = GaussianNB()
clf.fit(X, Y)
print(clf.score(X, Y))
print(clf.score(X_test, Y_test))
