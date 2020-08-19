import matplotlib.pyplot as plt
import numpy as np


labels = ['MCAR 40%', 'MCAR 50%', 'MCAR 60%', 'MCAR 70%', 'MCAR 80%']
# robust_inference_count = [12, 10, 25, 47, 49]
# missforest_count = [37, 39, 24, 2, 0]

robust_inference_count = [12, 10, 25, 47, 49]
missforest_count = [36, 39, 24, 2, 0]
mice = [1, 0, 0, 0, 0]

# robust_inference_count = [19, 20, 21, 22, 23]
# missforest_count = [10, 9, 10, 11, 10]
# mice = [4, 4, 2, 0, 0]

#robust_inference_count = [29, 31, 32, 35, 37]
#missforest_count = [27, 26, 25, 22, 20]
#mice = [1, 0, 0, 0, 0]

# robust_inference_count = [6, 9, 9, 11, 13]
# missforest_count = [14, 12, 13, 12, 10]
# mice = [3, 2, 1, 0, 0]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, robust_inference_count, width, label='Robust LR')
# rects2 = ax.bar(x + width/2, missforest_count, width, label='MissForest')

rects1 = ax.bar(x - width, robust_inference_count, width, label='Robust LR')
rects2 = ax.bar(x, missforest_count, width, label='MissForest')
rects3 = ax.bar(x + width, mice, width, label='MICE')


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of Wins')
# ax.set_title('Comparison of MissForest, MICE and Robust Learning on Parkinson Data Set')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


fig.tight_layout()

plt.show()
