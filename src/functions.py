# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.constants import (
    DATA_DIR, FIGURE_DIR, README_FILE, BATCH_IDS, N_BATCHES, DOWNTIME_DAYS,
    FEATURES, COLS_INITIAL, COLS_BATCH,
)

def culture_cost(glucose, glutamine, vcd):
    """Culture cost [$/L], with vcd in cells/mL (not millions)."""
    return 10.0 * np.asarray(glucose) + 20.0 * np.asarray(glutamine) + 0.0005 * np.asarray(vcd)


def economic_metrics(mab, cost, days):
    """Profit and yield include exactly one day of turnaround downtime."""
    value = 1000.0 * np.asarray(mab)
    space_time_profit = (value - np.asarray(cost)) / (np.asarray(days) + DOWNTIME_DAYS)
    space_time_yield = np.asarray(mab) / (np.asarray(days) + DOWNTIME_DAYS)
    return value, space_time_profit, space_time_yield


def find_batch_file(batch_id):
    """Accept both batch_####.csv and batches_####.csv for the stated run IDs."""
    candidates = [os.path.join(DATA_DIR, f"batch_{batch_id:04d}.csv"),
                  os.path.join(DATA_DIR, f"batches_{batch_id:04d}.csv")]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def ensure_batch_files():
    """Require all 30 batch CSV files locally; never access the internet."""
    missing = [i for i in BATCH_IDS if find_batch_file(i) is None]
    if missing:
        filenames = ", ".join(f"{i:04d}" for i in missing)
        raise FileNotFoundError(
            f"Missing batch CSV files for IDs: {filenames}. "
            "Place batch_####.csv or batches_####.csv files in datasets/ "
            "alongside initial_conditions.csv and rerun main.py. "
            "No files are downloaded automatically."
        )

def load_data():
    initial = pd.read_csv(os.path.join(DATA_DIR, "initial_conditions.csv"))
    if list(initial.columns) != ["Batch ID"] + COLS_INITIAL:
        raise ValueError("Unexpected columns in initial_conditions.csv")
    if initial["Batch ID"].duplicated().any():
        raise ValueError("Repeated Batch IDs in initial_conditions.csv")

    # Batch filenames are the experiment IDs, not a renumbered manifest.
    batch_ids = list(BATCH_IDS)
    if not set(batch_ids).issubset(set(initial["Batch ID"])):
        raise ValueError("Initial conditions are missing one or more of the 30 selected batch IDs")
    ensure_batch_files()

    design = initial.loc[initial["Batch ID"].isin(batch_ids)].copy()
    design = design.sort_values("Batch ID").reset_index(drop=True)
    series = []
    for batch_id in batch_ids:
        path = find_batch_file(batch_id)
        if path is None:
            raise FileNotFoundError(f"Missing data for batch {batch_id:04d}")
        df = pd.read_csv(path)
        if list(df.columns) != COLS_BATCH:
            raise ValueError(f"Unexpected column names in {os.path.basename(path)}: {list(df.columns)}")
        if len(df) < 6 or df[COLS_BATCH].isna().any().any():
            raise ValueError(f"Missing values or too few samples in {os.path.basename(path)}")
        df = df.sort_values("Time [h]").reset_index(drop=True)
        if df["Time [h]"].duplicated().any() or not np.all(np.diff(df["Time [h]"]) > 0):
            raise ValueError(f"Invalid time stamps in {os.path.basename(path)}")
        r = design.loc[design["Batch ID"] == batch_id].iloc[0]
        df["Batch ID"] = batch_id
        df[FEATURES[0]] = float(r["Glucose [mM]"])
        df[FEATURES[1]] = float(r["Glutamine [mM]"])
        df[FEATURES[2]] = float(r["VCD [cells/mL]"]) / 1e6
        df[FEATURES[3]] = df["Time [h]"] / 24.0
        df["Culture cost [$/L]"] = culture_cost(r["Glucose [mM]"], r["Glutamine [mM]"], r["VCD [cells/mL]"])
        df["Condition group"] = (f"{float(r['Glucose [mM]']):g}_"
                                 f"{float(r['Glutamine [mM]']):g}_"
                                 f"{float(r['VCD [cells/mL]']):g}")
        series.append(df)

    out = pd.concat(series, ignore_index=True)
    common_times = set(series[0]["Time [h]"].to_numpy())
    for s in series[1:]:
        common_times.intersection_update(s["Time [h]"].to_numpy())
    times = np.array(sorted(t for t in common_times if t > 0), dtype=float)
    if len(times) < 5:
        raise ValueError("Too few common positive harvest time points across the 30 batches")
    out = out.loc[out["Time [h]"].isin(times)].reset_index(drop=True)
    groups = out["Condition group"].nunique()
    if groups < 5:
        raise ValueError("At least five distinct starting-condition groups are needed for grouped CV")
    return design, initial, out, times


def evaluate_models(df):
    x = df[FEATURES].to_numpy(dtype=float)
    y = df["mAb [mg/L]"].to_numpy(dtype=float)
    groups = df["Condition group"].to_numpy()
    cv = GroupKFold(n_splits=5)

    candidates = {
        "Polynomial ridge": (
            make_pipeline(StandardScaler(), PolynomialFeatures(include_bias=False), Ridge()),
            {"polynomialfeatures__degree": [1, 2, 3], "ridge__alpha": [0.1, 1.0, 10.0]}),
        "kNN": (
            make_pipeline(StandardScaler(), KNeighborsRegressor()),
            {"kneighborsregressor__n_neighbors": [5, 15, 30],
             "kneighborsregressor__weights": ["uniform", "distance"]}),
    }
    records = []
    fitted = {}
    for name, (pipeline, params) in candidates.items():
        search = GridSearchCV(pipeline, param_grid=params, cv=cv, scoring="neg_root_mean_squared_error",
                              return_train_score=True, n_jobs=1)
        search.fit(x, y, groups=groups)
        i_best = int(search.best_index_)
        records.append({
            "Model": name,
            "CV train RMSE [mg/L]": -float(search.cv_results_["mean_train_score"][i_best]),
            "CV validation RMSE [mg/L]": -float(search.best_score_),
            "CV SD [mg/L]": float(search.cv_results_["std_test_score"][i_best]),
            "Hyperparameters": str(search.best_params_),
        })
        fitted[name] = search.best_estimator_

    comparison = pd.DataFrame(records).sort_values("CV validation RMSE [mg/L]").reset_index(drop=True)
    winner = comparison.at[0, "Model"]
    model = fitted[winner]
    # Out-of-fold predictions: folds keep identical starting conditions together.
    y_cv = cross_val_predict(clone(model), x, y, cv=cv, groups=groups, n_jobs=1)
    out = df.copy()
    out["CV prediction [mg/L]"] = y_cv
    out["CV residual [mg/L]"] = y - y_cv
    out["CV squared error [mg/L^2]"] = (y - y_cv) ** 2
    comparison["Selected"] = comparison["Model"].eq(winner)
    return model, comparison, out, float(np.sqrt(mean_squared_error(y, y_cv))), float(r2_score(y, y_cv))


def make_grid(design, times):
    """Predict only initial-condition combinations measured in the 30 selected runs.

    This avoids optimizing over unsupported glucose/glutamine/VCD combinations.
    """
    conditions = design[COLS_INITIAL].drop_duplicates().to_numpy(dtype=float)
    n_times = len(times)
    expanded = np.repeat(conditions, n_times, axis=0)
    harvest_hours = np.tile(times, len(conditions))
    return pd.DataFrame({
        FEATURES[0]: expanded[:, 0], FEATURES[1]: expanded[:, 1],
        FEATURES[2]: expanded[:, 2] / 1e6, FEATURES[3]: harvest_hours / 24.0,
        "Harvest time [h]": harvest_hours,
        "Initial VCD [cells/mL]": expanded[:, 2],
    })


def optimize(model, design, times):
    grid = make_grid(design, times)
    grid["Predicted mAb [mg/L]"] = np.maximum(0, model.predict(grid[FEATURES].to_numpy(dtype=float)))
    grid["Culture cost [$/L]"] = culture_cost(grid[FEATURES[0]], grid[FEATURES[1]], grid["Initial VCD [cells/mL]"])
    value, profit, yield_ = economic_metrics(grid["Predicted mAb [mg/L]"], grid["Culture cost [$/L]"], grid[FEATURES[3]])
    grid["Predicted value [$/L]"] = value
    grid["Predicted space-time profit [$/L/d]"] = profit
    grid["Predicted space-time yield [mg/L/d]"] = yield_
    best_i = int(grid["Predicted space-time profit [$/L/d]"].idxmax())
    best = grid.loc[best_i].copy()
    return grid, best


def intervals_by_time(df_cv, times):
    """Empirical marginal 95% prediction limits from held-out condition residuals."""
    records = []
    for t in times:
        residuals = df_cv.loc[df_cv["Time [h]"] == t, "CV residual [mg/L]"].to_numpy()
        if len(residuals) < 20:
            raise ValueError("Insufficient held-out residuals to estimate time-specific intervals.")
        lo, hi = np.quantile(residuals, [0.025, 0.975])
        records.append({"Harvest time [h]": t, "Residual 2.5% [mg/L]": lo, "Residual 97.5% [mg/L]": hi})
    return pd.DataFrame(records)


def add_intervals(df, residual_limits):
    out = df.merge(residual_limits, on="Harvest time [h]", how="left", validate="many_to_one")
    for lim, suffix in [("Residual 2.5% [mg/L]", "lower"), ("Residual 97.5% [mg/L]", "upper")]:
        mab = np.maximum(0.0, out["Predicted mAb [mg/L]"] + out[lim])
        val, profit, yield_ = economic_metrics(mab, out["Culture cost [$/L]"], out[FEATURES[3]])
        out[f"mAb PI {suffix} [mg/L]"] = mab
        out[f"Value PI {suffix} [$/L]"] = val
        out[f"Profit PI {suffix} [$/L/d]"] = profit
        out[f"Yield PI {suffix} [mg/L/d]"] = yield_
    return out


def factor_sensitivity(model, design, best):
    """One-factor predictions within levels represented in the supplied 30 batches."""
    records = []
    measured = design[COLS_INITIAL].drop_duplicates()
    for i, (factor, col) in enumerate(zip(FEATURES[:3], COLS_INITIAL)):
        for original_value in np.sort(measured[col].unique()):
            features = best[FEATURES].copy()
            x_value = float(original_value / 1e6) if i == 2 else float(original_value)
            features[factor] = x_value
            x = np.asarray([features.to_numpy(dtype=float)])
            mab = max(0.0, float(model.predict(x)[0]))
            vcd = float(features[FEATURES[2]]) * 1e6
            cost = culture_cost(features[FEATURES[0]], features[FEATURES[1]], vcd)
            _, profit, yield_ = economic_metrics(mab, cost, features[FEATURES[3]])
            condition = np.array([features[FEATURES[0]], features[FEATURES[1]], vcd])
            observed = np.isclose(measured.to_numpy(dtype=float), condition).all(axis=1).any()
            records.append({"Factor": factor, "Factor value": x_value,
                            "Profit [$/L/d]": float(profit),
                            "Yield [mg/L/d]": float(yield_),
                            "Starting conditions observed": bool(observed)})
    return pd.DataFrame(records)


def make_figures(comparison, cv_df, grid, best, sensitivities):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140, "savefig.dpi": 200})

    fig, ax = plt.subplots(figsize=(7.1, 4.3), layout="constrained")
    ax.bar(comparison["Model"], comparison["CV validation RMSE [mg/L]"], color=["tab:blue", "tab:orange"])
    ax.set_ylabel("Grouped CV RMSE [mg/L]")
    ax.set_title("Generalization across unseen initial conditions")
    fig.savefig(os.path.join(FIGURE_DIR, "01_model_comparison.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 5.3), layout="constrained")
    ax.scatter(cv_df["mAb [mg/L]"], cv_df["CV prediction [mg/L]"], s=8, alpha=0.22)
    a = min(cv_df["mAb [mg/L]"].min(), cv_df["CV prediction [mg/L]"].min())
    b = max(cv_df["mAb [mg/L]"].max(), cv_df["CV prediction [mg/L]"].max())
    ax.plot([a, b], [a, b], "k--", lw=1, label="Perfect prediction")
    ax.set(xlabel="Observed mAb [mg/L]", ylabel="Held-out prediction [mg/L]", title="Out-of-fold predictions")
    ax.legend()
    fig.savefig(os.path.join(FIGURE_DIR, "02_cv_parity.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    for batch_id in cv_df["Batch ID"].unique()[::5]:
        d = cv_df.loc[cv_df["Batch ID"] == batch_id]
        ax.plot(d[FEATURES[3]], d["mAb [mg/L]"], lw=1.2, alpha=0.85, label=f"Batch {batch_id:04d}")
    ax.set(xlabel="Culture time [d]", ylabel="mAb [mg/L]", title="Representative production trajectories")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(os.path.join(FIGURE_DIR, "03_batch_trajectories.png"))
    plt.close(fig)

    same = np.ones(len(grid), dtype=bool)
    for factor in FEATURES[:3]:
        same &= np.isclose(grid[factor], best[factor])
    curve = grid.loc[same].sort_values(FEATURES[3])
    fig, ax = plt.subplots(figsize=(7.1, 4.5), layout="constrained")
    ax.plot(curve[FEATURES[3]], curve["Predicted space-time profit [$/L/d]"], color="tab:blue", lw=1.8)
    ax.fill_between(curve[FEATURES[3]].to_numpy(), curve["Profit PI lower [$/L/d]"].to_numpy(),
                    curve["Profit PI upper [$/L/d]"].to_numpy(), alpha=0.22, color="tab:blue", label="Approx. 95% PI")
    ax.axvline(best[FEATURES[3]], ls="--", color="tab:red", label="Optimal harvest")
    ax.set(xlabel="Harvest time [d]", ylabel="Space-time profit [$/L/d]", title="Harvest timing at optimal starting conditions")
    ax.legend()
    fig.savefig(os.path.join(FIGURE_DIR, "04_optimal_harvest.png"))
    plt.close(fig)

    # Display only glucose/glutamine pairs actually represented in the design.
    # Each heatmap cell uses the best profit among its observed VCDs and harvest times.
    heat = (grid.groupby([FEATURES[1], FEATURES[0]])
            ["Predicted space-time profit [$/L/d]"].max().unstack(FEATURES[0]))
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    values = np.ma.masked_invalid(heat.to_numpy(dtype=float))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(heat.columns)), labels=[f"{a:g}" for a in heat.columns])
    ax.set_yticks(np.arange(len(heat.index)), labels=[f"{a:g}" for a in heat.index])
    ax.set(xlabel="Initial glucose [mM]", ylabel="Initial glutamine [mM]",
           title="Best predicted profit by observed nutrient pair")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.ma.is_masked(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", color="white", fontsize=9,
                        bbox={"facecolor": "black", "alpha": 0.27, "edgecolor": "none", "pad": 1})
    fig.colorbar(im, ax=ax, label="Max predicted profit among measured VCDs/times [$/L/d]")
    fig.savefig(os.path.join(FIGURE_DIR, "05_glucose_glutamine_heatmap.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), layout="constrained")
    for ax, factor in zip(axes, FEATURES[:3]):
        d = sensitivities.loc[sensitivities["Factor"] == factor]
        x = d["Factor value"].to_numpy()
        if factor == FEATURES[2]:
            xlabel = "Initial VCD [million cells/mL]"
        else:
            xlabel = factor
        ax.plot(x, d["Profit [$/L/d]"], marker="o", color="tab:blue")
        ax.axvline(best[factor], linestyle="--", color="tab:red")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Space-time profit [$/L/d]")
        ax.grid(alpha=0.2)
    fig.savefig(os.path.join(FIGURE_DIR, "06_factor_sensitivity.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4), layout="constrained")
    residual_summary = cv_df.groupby("Time [h]")["CV residual [mg/L]"].agg(["mean", "std"]).reset_index()
    time_days = residual_summary["Time [h]"].to_numpy() / 24
    ax.plot(time_days, residual_summary["mean"], color="tab:blue", label="Mean residual")
    ax.fill_between(time_days, (residual_summary["mean"] - residual_summary["std"]).to_numpy(),
                    (residual_summary["mean"] + residual_summary["std"]).to_numpy(), alpha=0.23, label="±1 SD")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Harvest time [d]", ylabel="Observed − predicted mAb [mg/L]", title="Prediction error versus time")
    ax.legend()
    fig.savefig(os.path.join(FIGURE_DIR, "07_cv_residuals.png"))
    plt.close(fig)


def update_readme(best, comparison, cv_rmse, cv_r2, sensitivities):
    readme = README_FILE
    if not os.path.isfile(readme):
        return
    with open(readme, "r", encoding="utf-8") as file:
        contents = file.read()
    start_tag, end_tag = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    if start_tag not in contents or end_tag not in contents:
        return
    model = comparison.loc[comparison["Selected"]].iloc[0]
    alternative = comparison.loc[~comparison["Selected"]].iloc[0]
    terms = [("Initial glucose", best[FEATURES[0]], "mM"),
             ("Initial glutamine", best[FEATURES[1]], "mM"),
             ("Initial viable cell density", best["Initial VCD [cells/mL]"], "cells/mL"),
             ("Harvest time", best[FEATURES[3]], "d"),
             ("Culture cost", best["Culture cost [$/L]"], "$/L"),
             ("Culture value at harvest", best["Predicted value [$/L]"], "$/L"),
             ("Space-time profit", best["Predicted space-time profit [$/L/d]"], "$/L/d"),
             ("Space-time mAb yield", best["Predicted space-time yield [mg/L/d]"], "mg/L/d")]
    lower = ["—", "—", "—", "—", "—", f"{best['Value PI lower [$/L]']:,.2f}",
             f"{best['Profit PI lower [$/L/d]']:,.2f}", f"{best['Yield PI lower [mg/L/d]']:.3f}"]
    upper = ["—", "—", "—", "—", "—", f"{best['Value PI upper [$/L]']:,.2f}",
             f"{best['Profit PI upper [$/L/d]']:,.2f}", f"{best['Yield PI upper [mg/L/d]']:.3f}"]
    rows = ["| Metric | Estimate | Approx. 95% PI lower | Approx. 95% PI upper |",
            "| --- | ---: | ---: | ---: |"]
    for (label, number, unit), lo, hi in zip(terms, lower, upper):
        fmt = f"{number:,.3f}" if unit == "mg/L/d" else f"{number:,.2f}" if unit != "cells/mL" else f"{number:,.0f}"
        rows.append(f"| {label} [{unit}] | {fmt} | {lo} | {hi} |")
    directions = []
    for factor in FEATURES[:3]:
        d = sensitivities.loc[sensitivities["Factor"] == factor].sort_values("Factor value")
        label = factor.replace(" [mM]", "").replace(" [10^6 cells/mL]", "")
        if len(d) < 2:
            directions.append(
                f"**{label}:** Only one initial level was represented in the 30 batches "
                f"({d['Factor value'].iloc[0]:g} in the model's input units). "
                "A response to changing this factor cannot be estimated from these runs."
            )
            continue
        low = float(d["Profit [$/L/d]"].iloc[0])
        high = float(d["Profit [$/L/d]"].iloc[-1])
        direction = "increased" if high > low else "decreased" if high < low else "did not change"
        extrapolated = (~d["Starting conditions observed"].astype(bool)).any()
        caution = (
            " Some factor settings combined with the recommended other inputs were not "
            "directly measured, so these conditional predictions may be less reliable."
            if extrapolated else " All points in this factor comparison were sampled."
        )
        directions.append(
            f"**{label}:** Over the levels represented in these batches, increasing "
            f"this factor {direction} predicted space-time profit from {low:,.2f} "
            f"to {high:,.2f} $/(L·d), with the other inputs and harvest time held "
            f"at their recommended values.{caution} These are model predictions, "
            "not established causal effects."
        )
    report = ("\n### Validated model performance and recommendation\n\n"
              f"The selected architecture was **{model['Model']}** with settings `{model['Hyperparameters']}`. "
              f"Its grouped cross-validation RMSE was **{model['CV validation RMSE [mg/L]']:.3f} mg/L** "
              f"(fold-score standard deviation {model['CV SD [mg/L]']:.3f} mg/L). "
              f"The alternative **{alternative['Model']}** obtained **{alternative['CV validation RMSE [mg/L]']:.3f} mg/L**. "
              f"The selected model's pooled out-of-fold RMSE was {cv_rmse:.3f} mg/L and "
              f"out-of-fold R² was {cv_r2:.3f}. These errors are measured on unseen initial-condition "
              "groups, while the model itself was refitted on all 30 runs for optimization.\n\n"
              + "\n".join(rows) + "\n\n"
              + "\n\n".join(directions) + "\n\n"
              "The values above optimize the **predicted mean** space-time profit on the supplied "
              "observed starting conditions and common measured harvest times, not a guarantee of the globally "
              "best physical operating point. The 95% limits are empirical, time-specific held-out "
              "residual quantiles transported to the recommended condition. They include process and "
              "forecast error represented by the selected runs, but do not fully account for model "
              "selection, selection of the maximum, or differing local variance.\n")
    pre = contents.split(start_tag, 1)[0]
    post = contents.split(end_tag, 1)[1]
    updated = pre + start_tag + "\n" + report + end_tag + post
    fig_start, fig_end = "<!-- FIGURE_STATUS_START -->", "<!-- FIGURE_STATUS_END -->"
    if fig_start in updated and fig_end in updated:
        before = updated.split(fig_start, 1)[0]
        after = updated.split(fig_end, 1)[1]
        updated = (before + fig_start + "\n"
                   + "All seven figures below were generated from the 30 batch datasets by `main.py`.\n"
                   + fig_end + after)
    with open(readme, "w", encoding="utf-8") as file:
        file.write(updated)