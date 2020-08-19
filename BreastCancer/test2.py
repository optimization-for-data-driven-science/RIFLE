from RobustImputer import RobustImputer
import sys


ind = sys.argv[1]

# arr = [400, 600, 800, 1000, 2000, 3000]
# data_ind = int(ind)
# n = arr[data_ind]
# print(n)
imputer = RobustImputer()

print(ind)
# imputer.read_and_scale('drive_MCAR80_' + str(n) + '.csv')
imputer.read_and_scale('drive_MCAR20_400_' + ind + '.csv')
imputer.estimate_confidence_intervals()
imputer.impute()

# imputer.write_to_csv('drive_imputed_MCAR80_' + str(n) + '.csv')
imputer.write_to_csv('drive_imputed_MCAR20_400_' + ind + '.csv')
