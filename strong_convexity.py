import random
import math
import numpy as np

m_list = []
for t in range(100000):

    error = 0.02
    correct = 0.98
    alpha1 = [correct * correct, correct*error, correct*error, error*error]
    alpha2 = [correct*error, correct * correct, error*error, correct*error]
    alpha3 = [correct*error, error*error, correct * correct, correct*error]
    alpha4 = [error*error, correct*error, correct*error, correct * correct]

    alpha1 = np.array(alpha1)
    alpha1 = alpha1[:, np.newaxis]

    alpha2 = np.array(alpha2)
    alpha2 = alpha2[:, np.newaxis]

    alpha3 = np.array(alpha3)
    alpha3 = alpha3[:, np.newaxis]

    alpha4 = np.array(alpha4)
    alpha4 = alpha4[:, np.newaxis]


    rho1 = [random.random(), random.random(), random.random(), random.random()]
    sum_rho = 0
    for item in rho1:
        sum_rho += item
    for i in range(len(rho1)):
        rho1[i] /= sum_rho

    rho1 = np.array(rho1)
    rho1 = rho1[:, np.newaxis]


    rho2 = [random.random(), random.random(), random.random(), random.random()]
    sum_rho = 0
    for item in rho2:
        sum_rho += item
    for i in range(len(rho2)):
        rho2[i] /= sum_rho

    rho2 = np.array(rho2)
    rho2 = rho2[:, np.newaxis]


    func1 = 0

    val = np.dot(rho1.T, alpha1)[0][0]
    func1 += val * math.log(val)

    val = np.dot(rho1.T, alpha2)[0][0]
    func1 += val * math.log(val)

    val = np.dot(rho1.T, alpha3)[0][0]
    func1 += val * math.log(val)

    val = np.dot(rho1.T, alpha4)[0][0]
    func1 += val * math.log(val)


    func2 = 0
    val = np.dot(rho2.T, alpha1)[0][0]
    func2 += val * math.log(val)

    val = np.dot(rho2.T, alpha2)[0][0]
    func2 += val * math.log(val)

    val = np.dot(rho2.T, alpha3)[0][0]
    func2 += val * math.log(val)

    val = np.dot(rho2.T, alpha4)[0][0]
    func2 += val * math.log(val)

    if func1 > func2:
        grad = math.log(np.dot(rho2.T, alpha1)[0][0]) * alpha1 + math.log(np.dot(rho2.T, alpha2)[0][0]) * alpha2 + math.log(np.dot(rho2.T, alpha3)[0][0]) * alpha3 + math.log(np.dot(rho2.T, alpha4)[0][0]) * alpha4 + alpha1 + alpha2 + alpha3 + alpha4
        approx = np.dot(grad.T, rho1 - rho2)[0][0]
        final_val = func1 - func2 - approx

    else:
        grad = math.log(np.dot(rho1.T, alpha1)[0][0]) * alpha1 + math.log(np.dot(rho1.T, alpha2)[0][0]) * alpha2 + math.log(np.dot(rho1.T, alpha3)[0][0]) * alpha3 + math.log(np.dot(rho1.T, alpha4)[0][0]) * alpha4 + alpha1 + alpha2 + alpha3 + alpha4
        approx = np.dot(grad.T, rho2 - rho1)[0][0]
        final_val = func2 - func1 - approx

    dist = np.linalg.norm(rho2-rho1, ord=1)
    m = final_val / (dist * dist)

    m_list.append(m)


print(min(m_list))
