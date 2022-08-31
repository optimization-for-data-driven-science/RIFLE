from RobustImputer import RobustImputer
import sys
import time


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
    start = time.time()
    run()
    end = time.time()
    print('Done in {:.4f} seconds'.format(end - start))