import pandas as pd
import csv

data = pd.read_csv('Sensorless_drive_diagnosis.txt', header=None, delimiter='\t')

dataset = []
for item in data.iterrows():

    x = item[1][0]
    y = []

    features = x.split()
    for num in range(len(features)-1):
        y.append(float(features[num]))

    dataset.append(y)

print(dataset)
with open("drive.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    for item in dataset:
        wr.writerow(item)
