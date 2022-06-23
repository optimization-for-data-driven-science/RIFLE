from RobustImputer import RobustImputer
import sys


def run():
    missing, imputed = sys.argv[1:3]
    imputer = RobustImputer()

    imputer.read_and_scale(missing)
    imputer.estimate_confidence_intervals()
    imputer.impute()
    imputer.write_to_csv(imputed)


# This guard is necessary to avoid creating subprocesses recursively.
# Without it a runtime error is generated, but there is likely a more clever way to do this
if __name__ == '__main__':
    run()