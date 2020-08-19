import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob

TrainData = pd.read_csv('blogData_train.csv', header=None)

sc = StandardScaler()

TrainData = sc.fit_transform(TrainData)

# Test Data
path = r'BlogFeedbackTest'
allFiles = glob.glob(path + "/*.csv")
test = pd.DataFrame()
for file_ in allFiles:
    test = test.append(pd.read_csv(file_, header=None))

test = sc.fit(test)
