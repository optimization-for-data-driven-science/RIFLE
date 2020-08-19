import pandas as pd
import numpy as np

imputed_data = pd.read_csv('MissForestImputed.csv')
original_data = pd.read_csv('training_data.csv')

# original_data.drop(['Po', 'At', 'Rn', 'H', 'F', 'Cl', 'Br', 'I', 'Nd'], axis=1, inplace=True)

original_data = original_data[original_data.columns[0:81]]
target = original_data[[original_data.columns[-1]]]
original_data = pd.concat([original_data, target], axis=1)

imputed_data = imputed_data[imputed_data.columns[0:81]]
target = imputed_data[[imputed_data.columns[-1]]]
imputed_data = pd.concat([imputed_data, target], axis=1)

print(imputed_data.shape)

mask = np.loadtxt('mask_pattern.csv', delimiter=',')
print(mask.shape)

t = []

for i in range(81, 158):
    t.append(i)

mask = np.delete(mask, t, axis=1)
print(mask.shape)
print(mask)

original_data.vars()
original_data = original_data.to_numpy()
imputed_data = imputed_data.to_numpy()

diff = original_data - imputed_data
print(diff)

print('----------------')
available_values = np.multiply(mask, diff)
print(available_values)

res = np.linalg.norm(available_values, 'fro')**2
mask_fro = np.linalg.norm(mask, 'fro')**2

print(res)
print(mask_fro)
print(res / mask_fro)
print(np.sqrt(res / mask_fro))
# --------------------------
imputed_data = pd.read_csv('NMARConductMice.csv')
original_data = pd.read_csv('training_data.csv')

imputed_data = imputed_data.drop(['Unnamed: 0'], axis=1)

# original_data.drop(['Po', 'At', 'Rn', 'H', 'F', 'Cl', 'Br', 'I', 'Nd'], axis=1, inplace=True)

original_data = original_data[original_data.columns[0:81]]
target = original_data[[original_data.columns[-1]]]
original_data = pd.concat([original_data, target], axis=1)

imputed_data = imputed_data[imputed_data.columns[0:81]]
target = imputed_data[[imputed_data.columns[-1]]]
imputed_data = pd.concat([imputed_data, target], axis=1)

print(imputed_data.shape)

mask = np.loadtxt('mask_pattern.csv', delimiter=',')
print(mask.shape)

t = []

for i in range(81, 158):
    t.append(i)

mask = np.delete(mask, t, axis=1)
print(mask.shape)
print(mask)

original_data = original_data.to_numpy()
imputed_data = imputed_data.to_numpy()

diff = original_data - imputed_data
print(diff)

print('----------------')
available_values = np.multiply(mask, diff)
print(available_values)

res = np.linalg.norm(available_values, 'fro')**2
mask_fro = np.linalg.norm(mask, 'fro')**2

print(mask.shape)
print(res)
print(mask_fro)
print(res / mask_fro)
print(np.sqrt(res / mask_fro))

# ---------------------------------
