import matplotlib.pyplot as plt

plt.style.use('bmh')

Number_of_estimations = [1, 2, 5, 10, 20, 30, 40, 50, 100]

# test_acc_avila_mcar = [62.49, 62.51, 63.13, 63.27, 63.32, 63.35, 63.36, 63.38, 63.43]
test_acc_magic_mcar = [76.7100, 76.7112, 76.7121, 76.7156, 76.7202, 76.725, 76.7304, 76.732, 76.7517]

fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()

# ax1.plot(lambda_train, train_eo, color='blue')
# ax2.plot(lambda_train, train_accuracy, color='r')

# ax1.plot(Number_of_estimations, test_acc_avila_mcar, color='r')
ax1.plot(Number_of_estimations, test_acc_magic_mcar, color='b')

plt.rc('xtick', labelsize=50)

plt.show()

