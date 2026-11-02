"""Constants and file paths used throughout the project."""

import os

# Set the project directory relative to the location of this file.
# os.path.join is used throughout, following the example course scripts.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "datasets")
FIGURE_DIR = os.path.join(PROJECT_DIR, "figures")
TABLE_DIR = os.path.join(PROJECT_DIR, "tables")

# 30 selected batches: ten starting conditions with three replicates each.
BATCH_IDS = (
    tuple(range(1, 7)) + tuple(range(22, 28)) + tuple(range(106, 109))
    + tuple(range(202, 211)) + tuple(range(301, 307))
)
DOWNTIME_DAYS = 1.0

FEATURES = [
    "Initial glucose [mM]",
    "Initial glutamine [mM]",
    "Initial VCD [10^6 cells/mL]",
    "Harvest time [d]",
]

COLS_INITIAL = ["Glucose [mM]", "Glutamine [mM]", "VCD [cells/mL]"]
COLS_BATCH = [
    "Time [h]", "Glucose [mM]", "Glutamine [mM]", "Lactate [mM]",
    "Ammonia [mM]", "VCD [cells/mL]", "TCD [cells/mL]", "mAb [mg/L]",
]
