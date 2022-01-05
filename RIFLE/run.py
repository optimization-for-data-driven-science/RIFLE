from RobustImputer import RobustImputer
import sys


missing, imputed = sys.argv[1:3]
imputer = RobustImputer()

imputer.read_and_scale(missing)
imputer.estimate_confidence_intervals()
imputer.impute()

imputer.write_to_csv(imputed)
