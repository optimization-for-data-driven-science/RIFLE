import random
import math
import numpy as np

m_list = []

alpha_num = 16


def new_alpha_val(old_alpha, k):
    new_alpha = np.zeros(old_alpha.shape)
    for ind in range(alpha_num):
        old_ind = (k - ind) % alpha_num
        new_alpha[ind][0] = old_alpha[old_ind][0]

    return new_alpha


e = 0.2
c = 0.8
# alpha1 = [c * c * c, c * c * e, c * e * e, e * e * e, c * e * e, c * c * e, c * e * e, c * c * e]
alpha1 = [c * c * c * c,
          c * c * c * e,
          c * c * e * e,
          c * e * e * e,
          e * e * e * e,
          c * e * e * e,
          c * c * e * e,
          c * c * c * e,
          c * e * e * e,
          c * c * e * e,
          c * e * e * e,
          c * c * e * e,
          c * c * c * e,
          c * c * e * e,
          c * c * c * e,
          c * c * e * e]

alpha1 = np.array(alpha1)
alpha1 = alpha1[:, np.newaxis]

alpha_list = [alpha1]

for i in range(alpha_num - 1):
    alpha_list.append(new_alpha_val(alpha_list[i], 2 * i + 1))

for t in range(100000):

    rho1 = []
    for j in range(alpha_num):
        rho1.append(random.random())

    sum_rho = 0
    for item in rho1:
        sum_rho += item
    for i in range(len(rho1)):
        rho1[i] /= sum_rho

    rho1 = np.array(rho1)
    rho1 = rho1[:, np.newaxis]

    rho2 = []
    for j in range(alpha_num):
        rho2.append(random.random())

    sum_rho = 0
    for item in rho2:
        sum_rho += item
    for i in range(len(rho2)):
        rho2[i] /= sum_rho

    rho2 = np.array(rho2)
    rho2 = rho2[:, np.newaxis]

    func1 = 0

    for i in range(alpha_num):
        val = np.dot(rho1.T, alpha_list[i])[0][0]
        func1 += val * math.log(val)

    func2 = 0
    for i in range(alpha_num):
        val = np.dot(rho2.T, alpha_list[i])[0][0]
        func2 += val * math.log(val)

    if func1 > func2:
        grad = np.zeros(rho2.shape)
        for i in range(alpha_num):
            grad += alpha_list[i] + math.log(np.dot(rho2.T, alpha_list[i])[0][0]) * alpha_list[i]

        approx = np.dot(grad.T, rho1 - rho2)[0][0]
        final_val = func1 - func2 - approx

    else:

        grad = np.zeros(rho1.shape)
        for i in range(alpha_num):
            grad += alpha_list[i] + math.log(np.dot(rho1.T, alpha_list[i])[0][0]) * alpha_list[i]

        approx = np.dot(grad.T, rho2 - rho1)[0][0]
        final_val = func2 - func1 - approx

    dist = np.linalg.norm(rho2 - rho1, ord=2)**2
    m = final_val / (dist)

    m_list.append(m)

print(min(m_list))
