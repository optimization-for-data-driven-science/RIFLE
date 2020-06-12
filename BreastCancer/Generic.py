import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import sqrt


def read_and_scale(filename):
    data = pd.read_csv(filename)

    sc = StandardScaler()
    sc.fit(data)

    cols = data.columns
    inds = data.index

    transformed = sc.transform(data)
    transformed_pandas_data = pd.DataFrame(transformed, columns=cols, index=inds)

    return data, transformed_pandas_data


def estimate_confidence_intervals(data):
    dimension = data.shape[1]
    confidence_matrix = np.zeros(shape=(dimension, dimension))

    sample_coeff = 1
    sampling_number = 30
    with_replacement = True
    # msk_gram = np.dot(mask_X.T, mask_X)

    cols = data.columns

    for i in range(dimension):
        for j in range(i, dimension):
            feature_i = cols[i]
            feature_j = cols[j]
            columns = data[[feature_i, feature_j]]
            intersections = columns[columns[[feature_i, feature_j]].notnull().all(axis=1)]

            intersection_num = len(intersections)

            sample_size = intersection_num // sample_coeff

            if sample_size < 2:
                max_vals = columns.max()
                max1 = max_vals[feature_i]
                max2 = max_vals[feature_j]
                confidence_matrix[i][j] = max1 * max2
                continue

            estimation_array = []
            for ind in range(sampling_number):
                current_sample = np.array(intersections.sample(n=sample_size, replace=with_replacement))
                with_replacement = False

                f1 = current_sample[:, 0]
                f2 = current_sample[:, 1]
                inner_prod = np.inner(f1, f2) / sample_size
                estimation_array.append(inner_prod)

            confidence_matrix[i][j] = np.std(estimation_array)

    for j in range(dimension):
        for i in range(j + 1, dimension):
            confidence_matrix[i][j] = confidence_matrix[j][i]

    return confidence_matrix


def impute_data(data, confidence_intervals, column_index, confidence_interval_coefficient=1):
    data_columns = data.columns
    y_column = data_columns[column_index]
    X = data.drop([y_column], axis=1)
    Y = data[[y_column]]

    y_list = data[y_column].to_numpy()
    Y_not_nan = np.nonzero(np.ones(y_list.shape) - np.isnan(y_list))[0]

    data_points = data.shape[0]
    validation_threshold = data.shape[1] // 10

    currentDelta = np.delete(confidence_intervals, column_index, 0)
    currentDelta = np.delete(currentDelta, column_index, 1)

    delta = confidence_intervals[column_index]
    delta = np.delete(delta, column_index, 0)
    delta = delta[:, np.newaxis]

    mask_X = X.isna()
    mask_X = mask_X.to_numpy()
    mask_X = np.ones(shape=mask_X.shape) - mask_X

    mask_Y_test = Y.isna()
    mask_Y_test = mask_Y_test.to_numpy()
    mask_Y = np.ones(shape=mask_Y_test.shape) - mask_Y_test

    mask_gram = np.dot(mask_X.T, mask_X)
    mask_gram = np.where(mask_gram == 0, 1, mask_gram)

    X = X.to_numpy()
    X = np.nan_to_num(X)

    Y = Y.to_numpy()
    Y = np.nan_to_num(Y)

    C = np.dot(X.T, X) / mask_gram
    b = np.dot(X.T, Y) / np.dot(mask_X.T, mask_Y)

    C_min = C - confidence_interval_coefficient * currentDelta
    C_max = C + confidence_interval_coefficient * currentDelta

    b_min = b - confidence_interval_coefficient * delta
    b_max = b + confidence_interval_coefficient * delta

    predicts = []

    for i in range(len(mask_X)):

        if not np.isnan(data[[y_column]].loc[i][0]):
            predicts.append(data[[y_column]].loc[i][0])
            continue

        row_i = mask_X[i]
        nonzeros = np.nonzero(row_i)[0]
        nonzeros = list(nonzeros)

        currentMask = mask_X[:, nonzeros]
        number_of_non_zeros = currentMask.shape[1] * np.ones(currentMask.shape[0]) - np.sum(currentMask, axis=1)

        size = 0
        counter = 0
        while size < data_points // 10:
            indices = np.argwhere(number_of_non_zeros < validation_threshold + counter)
            indices = np.reshape(indices, (indices.shape[0],))
            size = len(indices)
            counter += 1

        validation_indices = list(set(indices) & set(Y_not_nan))

        currentC = C[nonzeros, :]
        currentC = currentC[:, nonzeros]

        currentB = b[nonzeros, :]
        lam_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]

        # Optimizing CurrentC and currentB
        X_validation = X[validation_indices, :]
        X_validation = X_validation[:, nonzeros]
        X_validation = np.nan_to_num(X_validation)
        Y_validation = Y[validation_indices, :]

        best_lam = 1
        best_rmse = 1000000
        for lam in lam_list:
            theta = np.dot(np.linalg.inv(currentC + lam * np.identity(currentC.shape[0])), currentB)
            Y_predicted = np.dot(X_validation, theta)
            mse = np.linalg.norm(Y_validation - Y_predicted) ** 2 / Y_validation.shape[0]
            rmse = sqrt(mse)
            if rmse < best_rmse:
                best_rmse = rmse
                best_lam = lam

        # theta = np.dot(np.linalg.inv(currentC + best_lam * np.identity(currentC.shape[0])), currentB)

        data_i = X[i]
        data_i = data_i[:, np.newaxis]
        data_i = data_i[nonzeros, :]

        step_size = best_lam / 100
        number_of_iterations = 2000

        currentCmin = C_min[nonzeros, :]
        currentCmin = currentCmin[:, nonzeros]

        currentCmax = C_max[nonzeros, :]
        currentCmax = currentCmax[:, nonzeros]

        currentBmin = b_min[nonzeros, :]
        currentBmax = b_max[nonzeros, :]
        ident = np.identity(currentCmax.shape[0])

        best_res = 9999
        theta = np.dot(np.linalg.inv(currentC + best_lam * ident), currentB)

        for k in range(number_of_iterations):
            currentC += step_size * np.dot(theta, theta.T)
            currentC = np.clip(currentC, currentCmin, currentCmax)

            currentB += -2 * step_size * theta
            currentB = np.clip(currentB, currentBmin, currentBmax)

            theta = np.dot(np.linalg.inv(currentC + best_lam * ident), currentB)
            """
            if k % 100 == 1:
                new_predictions = np.dot(X_validation, theta)
                mse = np.linalg.norm(new_predictions - Y_validation) ** 2 / Y_validation.shape[0]
                if sqrt(mse) < best_res:
                    best_res = sqrt(mse)
                    bestB = currentB
                    bestC = currentC
            """

        theta = np.dot(np.linalg.inv(currentC + best_lam * np.identity(currentC.shape[0])), currentB)

        y_predict = np.dot(data_i.T, theta)
        # y_with_confidence = y_predict[0][0] * sqrt(sc_y.var_[0]) + sc_y.mean_[0]

        predicts.append(y_predict[0][0])

    return predicts


original_data, transformed_data = read_and_scale('missing_BC_MCAR30_1.csv')

data_cols = original_data.columns
standard_deviations = original_data.std()
means = original_data.mean()

Delta = estimate_confidence_intervals(transformed_data)

for column_index in range(original_data.shape[1]):
    predictions = impute_data(transformed_data, Delta, column_index)
    predictions = [x * standard_deviations[column_index] + means[column_index] for x in predictions]

    original_data[data_cols[column_index]] = predictions

original_data.to_csv("output.csv", index=False)