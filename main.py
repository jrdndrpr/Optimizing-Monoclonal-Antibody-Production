"""Find operating conditions that maximize mAb space-time profit."""

import sys

from src.classes import MonoclonalAntibodyOptimizer


if __name__ == "__main__":
    try:
        monitor = MonoclonalAntibodyOptimizer()

        # ==========================
        #   Load the Batch Data
        # ==========================
        monitor.load_batches()

        # ==========================
        #   Compare Regression Models
        # ==========================
        monitor.train_models()

        # ==========================
        #   Optimize Process Economics
        # ==========================
        monitor.find_optimum()

        # ==========================
        #   Export Figures and Tables
        # ==========================
        monitor.export_results()
        monitor.print_summary()

    except (FileNotFoundError, ValueError) as error:
        print(f"\nData error: {error}")
        sys.exit(1)
