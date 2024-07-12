import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import sqrt


class RobustImputer:

    def __init__(self):
        self.interval_length_constant = 1
        self.bootstrap_proportion = 1
        self.number_of_bootstrap_estimations = 30
        self.with_replacement = True
        self.validation_data_proportion = 0.1
        self.data = None
        self.transformed_data = None
        self.confidence_matrix = None
        self.imputed_data = None
        self.sc = None
        self.untouched = None
        self.mask1 = None
        self.mask0 = None
        self.remove_list = None

    def read_and_scale(self, filename):
        self.data = pd.read_csv(filename)

        remove_list = []
        columns = self.data.columns
        nan_count = self.data.shape[0] - self.data.isna().sum()
        # print(nan_count)
        for i in range(len(nan_count)):
            # print(columns[i], nan_count[i])
            if nan_count[i] < 5:
                remove_list.append(columns[i])
        #print(1/0)
        # for item in columns:
        #     current_col = self.data[item]
        #     if current_col.isnull().all():
        #        remove_list.append(item)

        self.data.drop(remove_list, axis=1, inplace=True)
        self.untouched = pd.read_csv(filename)

        self.untouched.drop(remove_list, axis=1, inplace=True)
        self.untouched = self.untouched.to_numpy()
        self.untouched = np.nan_to_num(self.untouched)

        if len(remove_list) > 0:
            print("These features are removed since they have less than 5 elements: ", remove_list)

        mask0 = self.data.isna()
        mask0 = mask0.to_numpy()
        mask1 = np.ones(shape=mask0.shape) - mask0  # 0 if Nan and 1 otherwise
        self.mask0 = mask0
        self.mask1 = mask1

        self.sc = StandardScaler()
        self.sc.fit(self.data)

        transformed = self.sc.transform(self.data)
        self.transformed_data = pd.DataFrame(transformed, columns=self.data.columns, index=self.data.index)

    def scale_data(self, data):
        self.data = data
        sc = StandardScaler()
        sc.fit(self.data)

        transformed = sc.transform(self.data)
        self.transformed_data = pd.DataFrame(transformed, columns=data.columns, index=data.index)

    def estimate_confidence_intervals(self):

        data = self.transformed_data
        dimension = data.shape[1]
        confidence_matrix = np.zeros(shape=(dimension, dimension))

        cols = data.columns

        # my_data = data['std_atomic_mass'].to_numpy()
        for i in range(dimension):
            for j in range(i, dimension):
                feature_i = cols[i]
                feature_j = cols[j]
                columns = data[[feature_i, feature_j]]
                intersections = columns[columns[[feature_i, feature_j]].notnull().all(axis=1)]

                intersection_num = len(intersections)

                sample_size = int(intersection_num * self.bootstrap_proportion)

                if sample_size < 2:
                    max_vals = columns.max()
                    max1 = max_vals[feature_i]
                    max2 = max_vals[feature_j]
                    # print("****")
                    # print("Max1:", max1)
                    # print("Max2:", max2)
                    # print(feature_i)
                    # print(feature_j)
                    # # print(max_vals)
                    # print(len(intersections))
                    # print("****")
                    confidence_matrix[i][j] = max1 * max2
                    continue

                estimation_array = []
                for _ in range(self.number_of_bootstrap_estimations):
                    current_sample = np.array(intersections.sample(n=sample_size, replace=self.with_replacement))
                    f1 = current_sample[:, 0]
                    f2 = current_sample[:, 1]
                    inner_prod = np.inner(f1, f2) / sample_size
                    estimation_array.append(inner_prod)

                confidence_matrix[i][j] = np.std(estimation_array)

        for j in range(dimension):
            for i in range(j + 1, dimension):
                confidence_matrix[i][j] = confidence_matrix[j][i]

        self.confidence_matrix = confidence_matrix

    def impute_data(self, column_index):
        data = self.transformed_data
        confidence_intervals = self.confidence_matrix

        data_columns = data.columns

        y_column = data_columns[column_index]
        X = data.drop([y_column], axis=1)
        Y = data[[y_column]]

        y_list = data[y_column].to_numpy()
        Y_not_nan = np.nonzero(np.ones(y_list.shape) - np.isnan(y_list))[0]

        data_points = data.shape[0]
        validation_threshold = int(self.validation_data_proportion * data.shape[1])

        currentDelta = np.delete(confidence_intervals, column_index, 0)
        currentDelta = np.delete(currentDelta, column_index, 1)

        delta = confidence_intervals[column_index]
        delta = np.delete(delta, column_index, 0)
        delta = delta[:, np.newaxis]

        mask_X = X.isna()
        mask_X = mask_X.to_numpy()
        mask_X = np.ones(shape=mask_X.shape) - mask_X  # 0 if Nan and 1 otherwise

        mask_Y_test = Y.isna()
        mask_Y_test = mask_Y_test.to_numpy()
        mask_Y = np.ones(shape=mask_Y_test.shape) - mask_Y_test

        mask_gram = np.dot(mask_X.T, mask_X)
        cov_gram = np.dot(mask_X.T, mask_Y)

        mask_gram = np.where(mask_gram == 0, 1, mask_gram)
        cov_gram = np.where(cov_gram == 0, 1, cov_gram)

        X = X.to_numpy()
        X = np.nan_to_num(X)

        Y = Y.to_numpy()
        Y = np.nan_to_num(Y)

        C = np.dot(X.T, X) / mask_gram
        b = np.dot(X.T, Y) / cov_gram

        C_min = C - self.interval_length_constant * currentDelta
        C_max = C + self.interval_length_constant * currentDelta

        b_min = b - self.interval_length_constant * delta
        b_max = b + self.interval_length_constant * delta

        predicts = []

        for i in range(len(mask_X)):
            # print(data[y_column])
            # print(data[y_column].iloc[i])
            # print(data[y_column].index[i])
            # print(1/0)
            if not np.isnan(data[y_column].iloc[i]):
                predicts.append(data[y_column].iloc[i])
                continue

            row_i = mask_X[i]
            nonzeros = np.nonzero(row_i)[0]
            nonzeros = list(nonzeros)

            currentMask = mask_X[:, nonzeros]
            number_of_non_zeros = currentMask.shape[1] * np.ones(currentMask.shape[0]) - np.sum(currentMask, axis=1)

            size = 0
            counter = 0
            indices = None
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

            theta = np.dot(np.linalg.inv(currentC + best_lam * ident), currentB)

            for k in range(number_of_iterations):
                currentC += step_size * np.dot(theta, theta.T)
                currentC = np.clip(currentC, currentCmin, currentCmax)

                currentB += -2 * step_size * theta
                currentB = np.clip(currentB, currentBmin, currentBmax)

                theta = np.dot(np.linalg.inv(currentC + best_lam * ident), currentB)

            y_predict = np.dot(data_i.T, theta)
            predicts.append(y_predict[0][0])

        return predicts

    def impute(self):
        original_data = self.data

        standard_deviations = original_data.std()
        means = original_data.mean()
        data_cols = original_data.columns

        for column_ind in range(original_data.shape[1]):
            print(data_cols[column_ind] + " is imputed.")
            predictions = self.impute_data(column_ind)
            predictions = [x * standard_deviations[column_ind] + means[column_ind] for x in predictions]

            original_data[data_cols[column_ind]] = predictions

        self.final_data = np.multiply(self.untouched, self.mask1) + np.multiply(original_data, self.mask0)
        # print(self.final_data)

    def write_to_csv(self, output_filename):
        self.final_data.to_csv(output_filename, index=False)
