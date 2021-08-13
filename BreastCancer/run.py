from RobustImputer import RobustImputer
import sys


ind = sys.argv[1]
imputer = RobustImputer()

print(ind)
# imputer.read_and_scale('drive_MCAR80_' + str(n) + '.csv')
imputer.read_and_scale('missing_BC_MCAR20.csv')
imputer.estimate_confidence_intervals()
imputer.impute()

# imputer.write_to_csv('drive_imputed_MCAR80_' + str(n) + '.csv')
imputer.write_to_csv('imputed_BC_MCAR20_instance' + str(ind) + '.csv')
