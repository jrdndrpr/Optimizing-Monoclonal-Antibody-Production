# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================

import os

from src.constants import FIGURE_DIR, TABLE_DIR, FEATURES
from src.functions import (
    load_data, evaluate_models, optimize, intervals_by_time, add_intervals,
    make_figures,
)

class MonoclonalAntibodyOptimizer:
    """Store results as the analysis progresses from raw data to a recommendation."""

    def load_batches(self):
        """Import and check the thirty time-series datasets."""
        self.design, self.initial, self.df, self.times = load_data()

    def train_models(self):
        """Use grouped cross-validation to compare and train regression models."""
        (self.model, self.comparison, self.cv_df,
         self.cv_rmse, self.cv_r2) = evaluate_models(self.df)

    def find_optimum(self):
        """Evaluate space-time profit at measured initial settings and harvest times."""
        self.grid, _ = optimize(self.model, self.design, self.times)
        self.residual_limits = intervals_by_time(self.cv_df, self.times)
        self.grid = add_intervals(self.grid, self.residual_limits)
        index = self.grid["Predicted space-time profit [$/L/d]"].idxmax()
        self.best = self.grid.loc[index].copy()

    def export_results(self):
        """Save the model comparison, optimization tables, and five figures."""
        os.makedirs(FIGURE_DIR, exist_ok=True)
        os.makedirs(TABLE_DIR, exist_ok=True)

        self.comparison.to_csv(os.path.join(TABLE_DIR, "model_comparison.csv"), index=False)
        columns = ["Batch ID", "Time [h]", "mAb [mg/L]", "CV prediction [mg/L]",
                   "CV residual [mg/L]", "Condition group"]
        self.cv_df.loc[:, columns].to_csv(
            os.path.join(TABLE_DIR, "held_out_predictions.csv"), index=False)
        self.residual_limits.to_csv(
            os.path.join(TABLE_DIR, "prediction_interval_residuals.csv"), index=False)
        self.design.to_csv(os.path.join(TABLE_DIR, "selected_batches.csv"), index=False)
        self.best.to_frame().T.to_csv(
            os.path.join(TABLE_DIR, "recommended_conditions.csv"), index=False)

        best_indices = self.grid.groupby(FEATURES[:3])["Predicted space-time profit [$/L/d]"].idxmax()
        top_conditions = self.grid.loc[best_indices].sort_values(
            "Predicted space-time profit [$/L/d]", ascending=False)
        top_conditions.to_csv(os.path.join(TABLE_DIR, "candidate_optima.csv"), index=False)

        make_figures(self.comparison, self.cv_df, self.grid, self.best)

    def print_summary(self):
        """Print a compact summary once all outputs have been generated."""
        selected = self.comparison.loc[self.comparison["Selected"], "Model"].iloc[0]
        print("\nSelected model:", selected)
        print("Model comparison:\n", self.comparison.to_string(index=False))
        print("\nRecommendation (approximate 95% prediction intervals):")
        keys = [FEATURES[0], FEATURES[1], "Initial VCD [cells/mL]", FEATURES[3],
                "Culture cost [$/L]", "Predicted value [$/L]",
                "Predicted space-time profit [$/L/d]", "Predicted space-time yield [mg/L/d]"]
        for key in keys:
            print(f"  {key}: {self.best[key]:,.4f}")
        print("\nFive figures saved in figures/ and six tables saved in tables/.")
