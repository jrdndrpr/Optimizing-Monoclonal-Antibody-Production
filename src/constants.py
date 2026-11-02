"""Constants and file paths used throughout the project."""

import os

# Set the project directory relative to the location of this file.
# os.path.join is used throughout, following the example course scripts.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "datasets")
FIGURE_DIR = os.path.join(PROJECT_DIR, "figures")
TABLE_DIR = os.path.join(PROJECT_DIR, "tables")

# Exactly 30 batch runs: 0100-0109, 0200-0209, and 0300-0309.
BATCH_IDS = tuple(range(100, 110)) + tuple(range(200, 210)) + tuple(range(300, 310))
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
