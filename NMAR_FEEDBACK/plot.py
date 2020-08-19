import matplotlib.pyplot as plt

# ['seaborn-ticks', 'ggplot', 'dark_background', 'bmh', 'seaborn-poster', 'seaborn-notebook', 'fast',
#  'seaborn', 'classic', 'Solarize_Light2', 'seaborn-dark', 'seaborn-pastel', 'seaborn-muted', '_classic_test',
#  'seaborn-paper', 'seaborn-colorblind', 'seaborn-bright', 'seaborn-talk', 'seaborn-dark-palette', 'tableau-colorblind10',
#  'seaborn-darkgrid', 'seaborn-whitegrid', 'fivethirtyeight', 'grayscale', 'seaborn-white', 'seaborn-deep']

# bmh
plt.style.use('bmh')

lambdas = ['0.01', '0.1', '1', '2', '5', '10', '20', '50', '100', '500', '1000', '2000', '5000', '10000']

missForest = [10.82708918, 5.686521325, 7.019262086, 8.472538681, 10.03013927, 10.94986946, 11.55493181, 11.44863094, 10.214612, 3.438792032, 1.030306415, 1.544944881, 1.857311021, 1.532912234]
noConfidence = [223.6498495, 212.722426, 13.56408606, 61.20165316, 0.972272734, 0.95708521, 0.954373153, 0.965763795, 0.977968056, 0.994240403, 0.996952461, 0.99830849, 0.999393313, 0.999664518]
confidence = [4.622516754, 3.3575, 1.5699, 1.17, 0.879, 0.884, 0.8922, 0.91, 0.912, 0.914, 0.914, 0.914, 0.914, 0.914]



lambdas = ['1', '2', '5', '10', '20', '50', '100', '500', '1000', '2000', '5000', '10000']

missForest = [7.019262086, 8.472538681, 10.03013927, 10.94986946, 11.55493181, 11.44863094, 10.214612, 3.438792032, 1.030306415, 1.544944881, 1.857311021, 1.532912234]
noConfidence = [13.56408606, 61.20165316, 0.972272734, 0.95708521, 0.954373153, 0.965763795, 0.977968056, 0.994240403, 0.996952461, 0.99830849, 0.999393313, 0.999664518]
confidence = [1.5699, 1.17, 0.879, 0.884, 0.8922, 0.91, 0.912, 0.914, 0.914, 0.914, 0.914, 0.914]

"""
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(lambdas, dp_violation, color='blue')

ax2.plot(lambdas, test_accuracy, color='r')

plt.rc('xtick', labelsize=50)
"""
plt.plot(lambdas, missForest)
plt.plot(lambdas, noConfidence)
plt.plot(lambdas, confidence)
plt.show()

