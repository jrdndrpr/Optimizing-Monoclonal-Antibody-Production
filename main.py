# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================

import sys

from src.classes import MonoclonalAntibodyOptimizer

if __name__ == "__main__":
    try:
        monitor = MonoclonalAntibodyOptimizer()
        monitor.load_batches()
        monitor.train_models()
        monitor.find_optimum()
        monitor.export_results()
        monitor.print_summary()

    except (FileNotFoundError, ValueError) as error:
        print(f"\nData error: {error}")
        sys.exit(1)