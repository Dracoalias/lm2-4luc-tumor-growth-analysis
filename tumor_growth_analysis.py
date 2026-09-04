#!/usr/bin/env python3
"""
LM2-4LUC Tumor Growth Model Analysis
====================================

A reproducible analysis pipeline for comparing exponential, logistic, and
Gompertz growth models independently for each mouse in the LM2-4LUC dataset.

Designed for the Mathematics: Applications and Interpretation HL IA workflow:
    raw data
      -> audit
      -> per-mouse time translation
      -> 3 nonlinear least-squares fits per mouse
      -> residuals / SSE / RMSE / R² / NRMSE
      -> optional AIC/AICc extension
      -> parameter-identifiability diagnostics
      -> per-mouse and cohort summaries
      -> plots

IMPORTANT METHODOLOGICAL CHOICES
--------------------------------
1. Mice are NEVER pooled into one growth trajectory.
2. Each mouse is fitted independently.
3. Time is translated separately for each mouse:
       t = Time - min(Time)
4. All three models minimize squared error in the ORIGINAL volume scale.
   The exponential model is NOT fitted by log-linear regression.
5. Positive model parameters are enforced through log-parameter optimization.
6. Logistic and Gompertz fits use deterministic multi-start optimization.
7. Zero-volume observations are not silently deleted. Use:
       --zero-policy retain
       --zero-policy exclude
       --zero-policy both
8. Very large fitted K values are FLAGGED, not automatically discarded.
9. AICc is included only as an optional extension. This script uses the number
   of fitted mean-function parameters as k (2 for exponential, 3 for logistic
   and Gompertz). If AICc is used in the IA, cite the chosen convention.

USAGE
-----
Typical full analysis:
    python tumor_growth_analysis.py LM2-4LUC.txt

Retain zeros only:
    python tumor_growth_analysis.py LM2-4LUC.txt --zero-policy retain

Quick test on mouse 53:
    python tumor_growth_analysis.py LM2-4LUC.txt --only-id 53 --zero-policy retain

Run without creating plots:
    python tumor_growth_analysis.py LM2-4LUC.txt --no-plots

Outputs are written under ./tumor_model_output by default.

Dependencies:
    numpy
    pandas
    scipy
    matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Matplotlib is imported lazily in plotting functions so --no-plots can be used
# even in a minimal/non-GUI environment.


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def exponential(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """V(t) = a exp(bt)."""
    z = np.clip(b * t, -700.0, 700.0)
    return a * np.exp(z)


def logistic(t: np.ndarray, K: float, A: float, r: float) -> np.ndarray:
    """V(t) = K / (1 + A exp(-rt))."""
    z = np.clip(-r * t, -700.0, 700.0)
    return K / (1.0 + A * np.exp(z))


def gompertz(t: np.ndarray, K: float, A: float, r: float) -> np.ndarray:
    """V(t) = K exp(-A exp(-rt))."""
    z = np.clip(-r * t, -700.0, 700.0)
    inner = A * np.exp(z)
    # inner is positive; clipping prevents numerical overflow/underflow.
    return K * np.exp(-np.clip(inner, 0.0, 700.0))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    func: Callable
    param_names: tuple[str, ...]
    k: int


MODELS: dict[str, ModelSpec] = {
    "Exponential": ModelSpec(
        "Exponential", exponential, ("a", "b"), 2
    ),
    "Logistic": ModelSpec(
        "Logistic", logistic, ("K", "A", "r"), 3
    ),
    "Gompertz": ModelSpec(
        "Gompertz", gompertz, ("K", "A", "r"), 3
    ),
}


# ---------------------------------------------------------------------------
# Data loading and audit
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["ID", "Time", "Observation"]


def load_data(path: Path) -> pd.DataFrame:
    """Load the supplied LM2-4LUC text file and validate its basic structure."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # The supplied file is tab-separated. Fall back to general whitespace.
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python")

    if not set(REQUIRED_COLUMNS).issubset(df.columns):
        # Retry with whitespace in case tabs/spaces were mixed.
        df = pd.read_csv(path, sep=r"\s+", engine="python")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # IDs should be integer-valued in this dataset.
    if not np.allclose(df["ID"], np.round(df["ID"])):
        raise ValueError("ID column contains non-integer values.")
    df["ID"] = df["ID"].astype(int)

    return df.sort_values(["ID", "Time"], kind="stable").reset_index(drop=True)


def audit_data(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Return audit summary, per-mouse counts, and zero-volume rows."""
    zero_rows = df.loc[df["Observation"] == 0, REQUIRED_COLUMNS].copy()
    counts = (
        df.groupby("ID")
        .agg(
            n=("Observation", "size"),
            first_time=("Time", "min"),
            last_time=("Time", "max"),
            min_observation=("Observation", "min"),
            max_observation=("Observation", "max"),
        )
        .reset_index()
    )
    counts["span_days"] = counts["last_time"] - counts["first_time"]

    count_distribution = (
        counts["n"].value_counts().sort_index().to_dict()
    )

    audit = {
        "rows": int(len(df)),
        "unique_mice": int(df["ID"].nunique()),
        "min_id": int(df["ID"].min()),
        "max_id": int(df["ID"].max()),
        "missing_values": {
            c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS
        },
        "zero_observations": int((df["Observation"] == 0).sum()),
        "negative_observations": int((df["Observation"] < 0).sum()),
        "duplicate_ID_Time_pairs": int(
            df.duplicated(["ID", "Time"]).sum()
        ),
        "measurements_per_mouse_min": int(counts["n"].min()),
        "measurements_per_mouse_max": int(counts["n"].max()),
        "measurement_count_distribution": {
            str(int(k)): int(v) for k, v in count_distribution.items()
        },
    }
    return audit, counts, zero_rows


def prepare_policy_data(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Apply an explicit zero-observation policy."""
    if policy == "retain":
        out = df.copy()
    elif policy == "exclude":
        out = df.loc[df["Observation"] != 0].copy()
    else:
        raise ValueError(f"Unknown zero policy: {policy}")

    if (out["Observation"] < 0).any():
        raise ValueError(
            "Negative tumor volumes are present. The script refuses to "
            "silently transform or remove them."
        )

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Starting values and nonlinear least-squares fitting
# ---------------------------------------------------------------------------

def _positive_first(y: np.ndarray) -> float:
    pos = y[y > 0]
    if len(pos):
        return float(pos[0])
    return 1.0


def _estimate_exponential_rate(t: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate a starting b from positive data only.
    This is ONLY an optimizer starting value; it is not the final regression.
    """
    mask = y > 0
    if mask.sum() >= 2:
        try:
            slope = float(np.polyfit(t[mask], np.log(y[mask]), 1)[0])
            return float(np.clip(slope, 0.005, 1.0))
        except Exception:
            pass
    return 0.10


def exponential_starts(t: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    y_ref = max(_positive_first(y), 1e-6)
    b_est = _estimate_exponential_rate(t, y)

    a_candidates = [0.5 * y_ref, y_ref, 1.5 * y_ref]
    b_candidates = [b_est, 0.03, 0.08, 0.15, 0.30]

    starts = []
    for a0 in a_candidates:
        for b0 in b_candidates:
            starts.append(np.array([max(a0, 1e-8), b0], dtype=float))
    return _deduplicate_starts(starts)


def saturating_starts(
    y: np.ndarray,
    model_name: str,
) -> list[np.ndarray]:
    """
    Deterministic multi-start guesses for Logistic/Gompertz.
    K starts above the largest observed value, but K is not constrained to
    remain above it during optimization.
    """
    y_max = max(float(np.max(y)), 1e-6)
    y_ref = max(_positive_first(y), 1e-6)

    k_ratios = [1.10, 1.50, 2.0, 5.0, 20.0, 100.0]
    r_candidates = [0.04, 0.10, 0.25]

    starts: list[np.ndarray] = []
    for ratio in k_ratios:
        K0 = max(y_max * ratio, 1.0)

        if model_name == "Logistic":
            A0 = max(K0 / y_ref - 1.0, 1e-3)
        elif model_name == "Gompertz":
            frac = min(max(y_ref / K0, 1e-300), 0.999999)
            A0 = max(-math.log(frac), 1e-3)
        else:
            raise ValueError(model_name)

        A0 = min(A0, 1e8)

        for r0 in r_candidates:
            starts.append(np.array([K0, A0, r0], dtype=float))

    return _deduplicate_starts(starts)


def _deduplicate_starts(starts: Iterable[np.ndarray]) -> list[np.ndarray]:
    seen = set()
    out = []
    for s in starts:
        key = tuple(np.round(np.log(np.asarray(s, dtype=float)), 10))
        if key not in seen:
            seen.add(key)
            out.append(np.asarray(s, dtype=float))
    return out


def parameter_bounds(
    model_name: str,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bounds in ORIGINAL parameter space.
    Optimization itself occurs in log-parameter space.
    """
    y_max = max(float(np.max(y)), 1e-6)

    if model_name == "Exponential":
        lower = np.array([1e-10, 1e-6], dtype=float)
        upper = np.array([max(y_max * 1e4, 10.0), 2.0], dtype=float)
    else:
        lower = np.array([1e-8, 1e-8, 1e-6], dtype=float)
        # Deliberately generous K upper bound so weakly identified asymptotes
        # can be detected and flagged rather than artificially suppressed.
        upper = np.array(
            [max(y_max * 1e6, 100.0), 1e10, 2.0],
            dtype=float,
        )

    return lower, upper


def _predict_from_log_params(
    spec: ModelSpec,
    t: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    params = np.exp(theta)
    return spec.func(t, *params)


def _approx_parameter_se(
    result,
    params: np.ndarray,
    sse: float,
    n: int,
    p: int,
) -> np.ndarray:
    """
    Approximate parameter standard errors from the least-squares Jacobian.

    The optimization parameters are logs of the physical parameters. The
    delta method converts SE(log parameter) to approximate SE(parameter).
    These SEs are diagnostics, not a substitute for a full uncertainty study.
    """
    if n <= p:
        return np.full(p, np.nan)

    try:
        J = np.asarray(result.jac, dtype=float)
        jtj_inv = np.linalg.pinv(J.T @ J)
        sigma2 = sse / (n - p)
        cov_theta = sigma2 * jtj_inv
        var_theta = np.diag(cov_theta)
        var_theta = np.where(var_theta >= 0, var_theta, np.nan)
        se_theta = np.sqrt(var_theta)
        return params * se_theta
    except Exception:
        return np.full(p, np.nan)


def fit_model_multistart(
    spec: ModelSpec,
    t: np.ndarray,
    y: np.ndarray,
    max_nfev: int,
) -> dict:
    """Fit one model using positive parameters and deterministic multi-starts."""
    lower, upper = parameter_bounds(spec.name, y)

    if spec.name == "Exponential":
        starts = exponential_starts(t, y)
    else:
        starts = saturating_starts(y, spec.name)

    log_lower = np.log(lower)
    log_upper = np.log(upper)

    best = None
    failures = 0

    for p0 in starts:
        # Ensure the starting point is strictly inside bounds.
        p0 = np.maximum(p0, lower * (1.0 + 1e-9))
        p0 = np.minimum(p0, upper * (1.0 - 1e-9))
        theta0 = np.log(p0)

        try:
            result = least_squares(
                fun=lambda theta: (
                    _predict_from_log_params(spec, t, theta) - y
                ),
                x0=theta0,
                bounds=(log_lower, log_upper),
                x_scale="jac",
                max_nfev=max_nfev,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )

            params = np.exp(result.x)
            yhat = spec.func(t, *params)

            if not np.all(np.isfinite(yhat)):
                failures += 1
                continue

            residuals = y - yhat
            sse = float(np.sum(residuals ** 2))

            if not np.isfinite(sse):
                failures += 1
                continue

            candidate = {
                "params": params,
                "yhat": yhat,
                "residuals": residuals,
                "sse": sse,
                "optimizer_success": bool(result.success),
                "optimizer_status": int(result.status),
                "optimizer_message": str(result.message),
                "optimizer_nfev": int(result.nfev),
                "optimizer_result": result,
                "lower": lower,
                "upper": upper,
            }

            if best is None or sse < best["sse"]:
                best = candidate

        except Exception:
            failures += 1

    if best is None:
        return {
            "success": False,
            "failure_count": failures,
            "start_count": len(starts),
        }

    best["success"] = True
    best["failure_count"] = failures
    best["start_count"] = len(starts)

    best["param_se"] = _approx_parameter_se(
        best["optimizer_result"],
        best["params"],
        best["sse"],
        n=len(y),
        p=spec.k,
    )

    # Remove the large scipy result object before saving/returning downstream.
    best.pop("optimizer_result", None)
    return best


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def fit_metrics(
    y: np.ndarray,
    yhat: np.ndarray,
    k: int,
) -> dict[str, float]:
    residuals = y - yhat
    sse = float(np.sum(residuals ** 2))
    n = len(y)

    rmse = float(np.sqrt(sse / n))
    mean_y = float(np.mean(y))

    if mean_y != 0:
        nrmse_mean = rmse / mean_y
    else:
        nrmse_mean = np.nan

    sst = float(np.sum((y - mean_y) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    # Gaussian least-squares AIC up to an additive constant.
    safe_mse = max(sse / n, np.finfo(float).tiny)
    aic = n * math.log(safe_mse) + 2 * k

    if n > k + 1:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = np.nan

    return {
        "SSE": sse,
        "RMSE": rmse,
        "NRMSE_mean": float(nrmse_mean),
        "R2": float(r2),
        "AIC_model_params": float(aic),
        "AICc_model_params": float(aicc),
        "mean_residual": float(np.mean(residuals)),
        "max_abs_residual": float(np.max(np.abs(residuals))),
    }


def equation_string(model_name: str, params: dict[str, float]) -> str:
    if model_name == "Exponential":
        return (
            f"V(t) = {params['a']:.8g} * exp({params['b']:.8g} * t)"
        )
    if model_name == "Logistic":
        return (
            f"V(t) = {params['K']:.8g} / "
            f"(1 + {params['A']:.8g} * exp(-{params['r']:.8g} * t))"
        )
    if model_name == "Gompertz":
        return (
            f"V(t) = {params['K']:.8g} * "
            f"exp(-{params['A']:.8g} * exp(-{params['r']:.8g} * t))"
        )
    return ""


# ---------------------------------------------------------------------------
# Per-mouse analysis
# ---------------------------------------------------------------------------

def analyse_mouse(
    mouse_df: pd.DataFrame,
    policy: str,
    max_nfev: int,
    k_flag_ratio: float,
) -> tuple[list[dict], list[dict], dict]:
    mouse_df = mouse_df.sort_values("Time", kind="stable").copy()

    mouse_id = int(mouse_df["ID"].iloc[0])
    absolute_time = mouse_df["Time"].to_numpy(dtype=float)
    y = mouse_df["Observation"].to_numpy(dtype=float)
    t = absolute_time - float(np.min(absolute_time))

    n = len(y)
    y_max = float(np.max(y))
    y_mean = float(np.mean(y))
    first_time = float(np.min(absolute_time))
    last_time = float(np.max(absolute_time))
    span_days = last_time - first_time

    result_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for model_name, spec in MODELS.items():
        fit = fit_model_multistart(
            spec=spec,
            t=t,
            y=y,
            max_nfev=max_nfev,
        )

        base = {
            "zero_policy": policy,
            "ID": mouse_id,
            "n": n,
            "first_time": first_time,
            "last_time": last_time,
            "span_days": span_days,
            "mean_observed": y_mean,
            "max_observed": y_max,
            "model": model_name,
            "parameter_count": spec.k,
            "fit_success": bool(fit.get("success", False)),
            "start_count": int(fit.get("start_count", 0)),
            "failed_starts": int(fit.get("failure_count", 0)),
        }

        if not fit.get("success", False):
            base.update({
                "equation": "",
                "SSE": np.nan,
                "RMSE": np.nan,
                "NRMSE_mean": np.nan,
                "R2": np.nan,
                "AIC_model_params": np.nan,
                "AICc_model_params": np.nan,
                "mean_residual": np.nan,
                "max_abs_residual": np.nan,
                "a": np.nan,
                "b": np.nan,
                "K": np.nan,
                "A": np.nan,
                "r": np.nan,
                "SE_a": np.nan,
                "SE_b": np.nan,
                "SE_K": np.nan,
                "SE_A": np.nan,
                "SE_r": np.nan,
                "K_to_max_observed": np.nan,
                "K_relative_SE": np.nan,
                "K_far_above_data_flag": False,
                "K_near_upper_bound_flag": False,
                "optimizer_success": False,
                "optimizer_status": np.nan,
                "optimizer_nfev": np.nan,
                "optimizer_message": "All starts failed",
            })
            result_rows.append(base)
            continue

        metrics = fit_metrics(y, fit["yhat"], spec.k)
        params = {
            name: float(value)
            for name, value in zip(spec.param_names, fit["params"])
        }
        param_se = {
            name: float(value)
            for name, value in zip(spec.param_names, fit["param_se"])
        }

        base.update(metrics)
        base["equation"] = equation_string(model_name, params)

        # Stable rectangular schema across all three models.
        for p in ["a", "b", "K", "A", "r"]:
            base[p] = params.get(p, np.nan)
            base[f"SE_{p}"] = param_se.get(p, np.nan)

        if model_name in ("Logistic", "Gompertz"):
            K = params["K"]
            se_K = param_se.get("K", np.nan)
            K_ratio = K / y_max if y_max > 0 else np.nan
            K_rel_se = se_K / K if K > 0 and np.isfinite(se_K) else np.nan
            upper_K = float(fit["upper"][0])

            base["K_to_max_observed"] = float(K_ratio)
            base["K_relative_SE"] = float(K_rel_se)
            base["K_far_above_data_flag"] = bool(
                np.isfinite(K_ratio) and K_ratio > k_flag_ratio
            )
            base["K_near_upper_bound_flag"] = bool(
                K >= 0.99 * upper_K
            )
        else:
            base["K_to_max_observed"] = np.nan
            base["K_relative_SE"] = np.nan
            base["K_far_above_data_flag"] = False
            base["K_near_upper_bound_flag"] = False

        base["optimizer_success"] = bool(fit["optimizer_success"])
        base["optimizer_status"] = int(fit["optimizer_status"])
        base["optimizer_nfev"] = int(fit["optimizer_nfev"])
        base["optimizer_message"] = fit["optimizer_message"]

        result_rows.append(base)

        for i in range(n):
            prediction_rows.append({
                "zero_policy": policy,
                "ID": mouse_id,
                "model": model_name,
                "absolute_time": float(absolute_time[i]),
                "t_since_first_measurement": float(t[i]),
                "observed": float(y[i]),
                "predicted": float(fit["yhat"][i]),
                "residual_observed_minus_predicted": float(
                    fit["residuals"][i]
                ),
            })

    mouse_meta = {
        "ID": mouse_id,
        "t": t,
        "absolute_time": absolute_time,
        "y": y,
        "result_rows": result_rows,
    }

    return result_rows, prediction_rows, mouse_meta


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _winner_from_group(
    group: pd.DataFrame,
    metric: str,
) -> tuple[str | None, float, float]:
    valid = group.loc[
        group["fit_success"] & group[metric].notna(),
        ["model", metric],
    ].sort_values(metric, kind="stable")

    if valid.empty:
        return None, np.nan, np.nan

    best_name = str(valid.iloc[0]["model"])
    best_value = float(valid.iloc[0][metric])

    if len(valid) >= 2:
        second = float(valid.iloc[1][metric])
        gap = second - best_value
    else:
        gap = np.nan

    return best_name, best_value, gap


def make_mouse_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for mouse_id, g in results.groupby("ID", sort=True):
        base = {
            "zero_policy": str(g["zero_policy"].iloc[0]),
            "ID": int(mouse_id),
            "n": int(g["n"].iloc[0]),
            "first_time": float(g["first_time"].iloc[0]),
            "last_time": float(g["last_time"].iloc[0]),
            "span_days": float(g["span_days"].iloc[0]),
            "mean_observed": float(g["mean_observed"].iloc[0]),
            "max_observed": float(g["max_observed"].iloc[0]),
        }

        rmse_winner, rmse_best, rmse_gap = _winner_from_group(g, "RMSE")
        aicc_winner, aicc_best, aicc_gap = _winner_from_group(
            g, "AICc_model_params"
        )

        base["best_RMSE_model"] = rmse_winner
        base["best_RMSE"] = rmse_best
        base["RMSE_gap_to_second_best"] = rmse_gap
        base["best_AICc_model_params_model"] = aicc_winner
        base["best_AICc_model_params"] = aicc_best
        base["AICc_gap_to_second_best"] = aicc_gap

        # Wide-form metrics for easy IA tables and comparisons.
        for _, row in g.iterrows():
            label = str(row["model"]).lower()
            base[f"{label}_RMSE"] = row["RMSE"]
            base[f"{label}_NRMSE_mean"] = row["NRMSE_mean"]
            base[f"{label}_R2"] = row["R2"]
            base[f"{label}_AICc_model_params"] = row[
                "AICc_model_params"
            ]

            if row["model"] in ("Logistic", "Gompertz"):
                base[f"{label}_K"] = row["K"]
                base[f"{label}_K_to_max_observed"] = row[
                    "K_to_max_observed"
                ]
                base[f"{label}_K_far_above_data_flag"] = row[
                    "K_far_above_data_flag"
                ]

        # Negative means Gompertz has lower RMSE and therefore fits better.
        if (
            "gompertz_RMSE" in base and
            "exponential_RMSE" in base
        ):
            base["Gompertz_minus_Exponential_RMSE"] = (
                base["gompertz_RMSE"] - base["exponential_RMSE"]
            )
        else:
            base["Gompertz_minus_Exponential_RMSE"] = np.nan

        if (
            "gompertz_RMSE" in base and
            "logistic_RMSE" in base
        ):
            base["Gompertz_minus_Logistic_RMSE"] = (
                base["gompertz_RMSE"] - base["logistic_RMSE"]
            )
        else:
            base["Gompertz_minus_Logistic_RMSE"] = np.nan

        rows.append(base)

    return pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)


def cohort_tables(
    mouse_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_mice = len(mouse_summary)

    rmse_counts = (
        mouse_summary["best_RMSE_model"]
        .value_counts(dropna=False)
        .rename_axis("model")
        .reset_index(name="mice")
    )
    rmse_counts["percentage"] = 100 * rmse_counts["mice"] / n_mice
    rmse_counts.insert(0, "criterion", "lowest_RMSE")

    aicc_counts = (
        mouse_summary["best_AICc_model_params_model"]
        .value_counts(dropna=False)
        .rename_axis("model")
        .reset_index(name="mice")
    )
    aicc_counts["percentage"] = 100 * aicc_counts["mice"] / n_mice
    aicc_counts.insert(
        0,
        "criterion",
        "lowest_AICc_model_params_convention",
    )

    cohort = pd.concat([rmse_counts, aicc_counts], ignore_index=True)

    by_n = (
        mouse_summary.groupby(["n", "best_RMSE_model"])
        .size()
        .rename("mice")
        .reset_index()
        .sort_values(["n", "best_RMSE_model"])
    )

    by_span = mouse_summary[
        [
            "ID",
            "n",
            "span_days",
            "best_RMSE_model",
            "Gompertz_minus_Exponential_RMSE",
            "Gompertz_minus_Logistic_RMSE",
        ]
    ].copy()

    return cohort, by_n, by_span


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mouse(
    mouse_meta: dict,
    results: pd.DataFrame,
    policy: str,
    plot_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    mouse_id = int(mouse_meta["ID"])
    t = np.asarray(mouse_meta["t"], dtype=float)
    y = np.asarray(mouse_meta["y"], dtype=float)

    g = results.loc[results["ID"] == mouse_id].copy()

    # 1) Observed data + all fitted curves
    plt.figure(figsize=(8, 5.5))
    plt.scatter(t, y, label="Observed", zorder=3)

    fine_t = np.linspace(float(np.min(t)), float(np.max(t)), 400)

    for _, row in g.iterrows():
        if not bool(row["fit_success"]):
            continue

        model_name = str(row["model"])
        if model_name == "Exponential":
            pred = exponential(fine_t, row["a"], row["b"])
        elif model_name == "Logistic":
            pred = logistic(fine_t, row["K"], row["A"], row["r"])
        else:
            pred = gompertz(fine_t, row["K"], row["A"], row["r"])

        plt.plot(fine_t, pred, label=model_name)

    plt.xlabel("Days since first recorded measurement, t")
    plt.ylabel("Tumor volume (mm³)")
    plt.title(f"Mouse {mouse_id}: observed data and fitted models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        plot_dir / f"mouse_{mouse_id:02d}_model_fits.png",
        dpi=180,
    )
    plt.close()

    # 2) Residuals
    plt.figure(figsize=(8, 5.5))
    for model_name, spec in MODELS.items():
        row = g.loc[g["model"] == model_name]
        if row.empty or not bool(row.iloc[0]["fit_success"]):
            continue

        row = row.iloc[0]
        if model_name == "Exponential":
            pred = exponential(t, row["a"], row["b"])
        elif model_name == "Logistic":
            pred = logistic(t, row["K"], row["A"], row["r"])
        else:
            pred = gompertz(t, row["K"], row["A"], row["r"])

        residual = y - pred
        plt.plot(t, residual, marker="o", label=model_name)

    plt.axhline(0, linewidth=1, linestyle="--")
    plt.xlabel("Days since first recorded measurement, t")
    plt.ylabel("Residual: observed − predicted (mm³)")
    plt.title(f"Mouse {mouse_id}: residuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        plot_dir / f"mouse_{mouse_id:02d}_residuals.png",
        dpi=180,
    )
    plt.close()


def plot_cohort(
    mouse_summary: pd.DataFrame,
    cohort: pd.DataFrame,
    by_n: pd.DataFrame,
    results: pd.DataFrame,
    plot_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    # 1) RMSE winner counts
    rmse_counts = cohort.loc[
        cohort["criterion"] == "lowest_RMSE"
    ].copy()

    plt.figure(figsize=(7, 5))
    plt.bar(rmse_counts["model"].astype(str), rmse_counts["mice"])
    plt.ylabel("Number of mice")
    plt.xlabel("Model with lowest RMSE")
    plt.title("Model wins across mice by RMSE")
    plt.tight_layout()
    plt.savefig(plot_dir / "cohort_RMSE_winner_counts.png", dpi=180)
    plt.close()

    # 2) Gompertz pairwise RMSE differences
    plt.figure(figsize=(9, 5.5))
    plt.plot(
        mouse_summary["ID"],
        mouse_summary["Gompertz_minus_Exponential_RMSE"],
        marker="o",
        linestyle="none",
        label="Gompertz − Exponential",
    )
    plt.plot(
        mouse_summary["ID"],
        mouse_summary["Gompertz_minus_Logistic_RMSE"],
        marker="x",
        linestyle="none",
        label="Gompertz − Logistic",
    )
    plt.axhline(0, linewidth=1, linestyle="--")
    plt.xlabel("Mouse ID")
    plt.ylabel("RMSE difference (mm³)")
    plt.title(
        "Gompertz RMSE differences\n"
        "Negative values favour Gompertz"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        plot_dir / "cohort_Gompertz_RMSE_differences.png",
        dpi=180,
    )
    plt.close()

    # 3) Winner counts by number of measurements
    if not by_n.empty:
        pivot = by_n.pivot(
            index="n",
            columns="best_RMSE_model",
            values="mice",
        ).fillna(0)

        plt.figure(figsize=(9, 5.5))
        ax = pivot.plot(kind="bar")
        ax.set_xlabel("Number of observations for a mouse")
        ax.set_ylabel("Number of mice")
        ax.set_title("RMSE winner by trajectory sample size")
        ax.legend(title="Best model")
        ax.figure.tight_layout()
        ax.figure.savefig(
            plot_dir / "cohort_RMSE_winner_by_n.png",
            dpi=180,
        )
        plt.close(ax.figure)

    # 4) K identifiability diagnostic
    sat = results.loc[
        results["model"].isin(["Logistic", "Gompertz"])
        & results["fit_success"]
        & results["K_to_max_observed"].notna()
    ].copy()

    if not sat.empty:
        plt.figure(figsize=(9, 5.5))
        for model_name in ["Logistic", "Gompertz"]:
            gg = sat.loc[sat["model"] == model_name]
            plt.plot(
                gg["ID"],
                gg["K_to_max_observed"],
                marker="o",
                linestyle="none",
                label=model_name,
            )

        plt.axhline(10, linewidth=1, linestyle="--")
        plt.yscale("log")
        plt.xlabel("Mouse ID")
        plt.ylabel("Fitted K / largest observed volume (log scale)")
        plt.title("Asymptote-identifiability diagnostic")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            plot_dir / "cohort_K_to_observed_max.png",
            dpi=180,
        )
        plt.close()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_notes(
    path: Path,
    audit: dict,
    policy: str,
    n_analysed: int,
    k_flag_ratio: float,
) -> None:
    text = f"""LM2-4LUC ANALYSIS NOTES
======================

ZERO POLICY
-----------
Policy used in this output folder: {policy}

The script does not infer what a recorded zero means. If the original source
does not explain it clearly, do not silently describe a zero as "missing" or
"not measurable". State the chosen rule and, ideally, compare retain/exclude
results as a sensitivity analysis.

RAW DATA AUDIT
--------------
Rows in raw file: {audit['rows']}
Unique mice in raw file: {audit['unique_mice']}
ID range: {audit['min_id']} to {audit['max_id']}
Recorded zero observations: {audit['zero_observations']}
Recorded negative observations: {audit['negative_observations']}
Duplicate (ID, Time) pairs: {audit['duplicate_ID_Time_pairs']}
Mice analysed in this run: {n_analysed}

MODEL FITTING
-------------
Each mouse is analysed independently.

For each mouse:
    t = Time - first recorded Time for that mouse

Models:
    Exponential: V(t) = a exp(bt)
    Logistic:    V(t) = K / (1 + A exp(-rt))
    Gompertz:    V(t) = K exp(-A exp(-rt))

All fits minimize:
    SSE = sum((observed - predicted)^2)

in the ORIGINAL tumor-volume scale.

The exponential model is NOT fitted using a log-transformed linear regression,
because doing so would minimize squared errors in log-volume rather than the
same squared-volume objective used by Logistic and Gompertz.

METRICS
-------
Residual:
    e_i = observed_i - predicted_i

RMSE:
    sqrt(SSE / n)

R²:
    1 - SSE/SST

NRMSE_mean:
    RMSE / mean(observed volume)

IMPORTANT:
For a given mouse all models use the same observations and n. Therefore RMSE
and R² are both monotonic functions of SSE and will rank the three models in
the same order. Reporting both can still communicate fit in different forms,
but they are not independent pieces of evidence for which model wins.

AIC/AICc EXTENSION
------------------
This script also outputs:
    AIC = n ln(SSE/n) + 2k
    AICc = AIC + 2k(k+1)/(n-k-1)

where this implementation defines k as the number of fitted mean-function
parameters:
    Exponential k = 2
    Logistic k = 3
    Gompertz k = 3

Some statistical conventions count the estimated error variance as an
additional parameter. If AICc appears in the IA, cite the exact convention
being used and apply it consistently. For that reason the output column is
explicitly named AICc_model_params.

K / ASYMPTOTE CAUTION
---------------------
The Logistic and Gompertz parameter K is an estimated upper asymptote.
The experimental data may stop before a long-term plateau is visible.

This program flags:
    K / max(observed volume) > {k_flag_ratio:g}

as "K_far_above_data_flag = True".

That threshold is ONLY a diagnostic flag, not a mathematical exclusion rule.
A very large K often means the observed interval does not identify the
asymptote well. It does not automatically mean the in-sample curve fit failed.

Suggested interpretation:
    distinguish "good descriptive fit over observed times"
    from "well-supported biological interpretation of K".

FILES
-----
model_results.csv
    One row per mouse-model fit. Contains parameters, errors, metrics, AICc,
    optimizer information, and K diagnostics.

predictions_and_residuals.csv
    One row per observation per fitted model.

mouse_summary.csv
    One row per mouse. Convenient for IA summary tables.

cohort_summary.csv
    Counts and percentages of model winners by RMSE and AICc.

RMSE_winners_by_measurement_count.csv
    Whether the winning model changes with the amount of data available.

RMSE_differences_and_span.csv
    Gompertz-vs-other RMSE differences alongside trajectory length.

plots/
    Per-mouse model-fit and residual plots plus cohort diagnostics.

IA PRESENTATION SUGGESTION
--------------------------
Use one mouse (for example ID 53) as the detailed worked mathematical case:
    equations
    regression parameters
    a sample residual calculation
    RMSE/R²
    residual plot
    calculus / growth-rate interpretation

Then use the all-mice Python analysis as a robustness/generalisation extension.
Do not put 198 regressions into the main body.
"""
    path.write_text(text, encoding="utf-8")


def save_focus_mouse(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    focus_id: int,
    out_dir: Path,
) -> None:
    focus_results = results.loc[results["ID"] == focus_id].copy()
    focus_predictions = predictions.loc[predictions["ID"] == focus_id].copy()

    if focus_results.empty:
        return

    focus_results.to_csv(
        out_dir / f"focus_mouse_{focus_id}_model_results.csv",
        index=False,
    )
    focus_predictions.to_csv(
        out_dir / f"focus_mouse_{focus_id}_predictions_residuals.csv",
        index=False,
    )


# ---------------------------------------------------------------------------
# Dataset-level pipeline
# ---------------------------------------------------------------------------

def analyse_policy(
    raw_df: pd.DataFrame,
    policy: str,
    base_output: Path,
    only_ids: list[int] | None,
    focus_id: int,
    make_plots: bool,
    max_nfev: int,
    k_flag_ratio: float,
    audit: dict,
) -> dict:
    analysis_df = prepare_policy_data(raw_df, policy)

    if only_ids:
        missing = sorted(set(only_ids) - set(analysis_df["ID"].unique()))
        if missing:
            warnings.warn(f"Requested IDs not found: {missing}")
        analysis_df = analysis_df.loc[
            analysis_df["ID"].isin(only_ids)
        ].copy()

    policy_dir = base_output / f"zero_policy_{policy}"
    policy_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = policy_dir / "plots"
    if make_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    all_predictions: list[dict] = []
    metas: list[dict] = []

    ids = sorted(analysis_df["ID"].unique())
    print(
        f"\n[{policy}] Analysing {len(ids)} mice "
        f"({len(ids) * len(MODELS)} model fits)..."
    )

    for index, mouse_id in enumerate(ids, start=1):
        mouse_df = analysis_df.loc[analysis_df["ID"] == mouse_id].copy()

        result_rows, prediction_rows, meta = analyse_mouse(
            mouse_df=mouse_df,
            policy=policy,
            max_nfev=max_nfev,
            k_flag_ratio=k_flag_ratio,
        )

        all_results.extend(result_rows)
        all_predictions.extend(prediction_rows)
        metas.append(meta)

        print(
            f"  [{index:>2}/{len(ids)}] mouse {mouse_id:>2} complete",
            end="\r",
            flush=True,
        )

    print()

    results = pd.DataFrame(all_results)
    predictions = pd.DataFrame(all_predictions)

    results.to_csv(policy_dir / "model_results.csv", index=False)
    predictions.to_csv(
        policy_dir / "predictions_and_residuals.csv",
        index=False,
    )

    mouse_summary = make_mouse_summary(results)
    mouse_summary.to_csv(policy_dir / "mouse_summary.csv", index=False)

    cohort, by_n, by_span = cohort_tables(mouse_summary)
    cohort.to_csv(policy_dir / "cohort_summary.csv", index=False)
    by_n.to_csv(
        policy_dir / "RMSE_winners_by_measurement_count.csv",
        index=False,
    )
    by_span.to_csv(
        policy_dir / "RMSE_differences_and_span.csv",
        index=False,
    )

    save_focus_mouse(
        results=results,
        predictions=predictions,
        focus_id=focus_id,
        out_dir=policy_dir,
    )

    write_notes(
        path=policy_dir / "analysis_notes.txt",
        audit=audit,
        policy=policy,
        n_analysed=len(ids),
        k_flag_ratio=k_flag_ratio,
    )

    if make_plots:
        print(f"[{policy}] Creating plots...")
        for meta in metas:
            plot_mouse(
                mouse_meta=meta,
                results=results,
                policy=policy,
                plot_dir=plot_dir,
            )
        plot_cohort(
            mouse_summary=mouse_summary,
            cohort=cohort,
            by_n=by_n,
            results=results,
            plot_dir=plot_dir,
        )

    # Human-readable terminal summary.
    print(f"\n[{policy}] RMSE winners:")
    rmse_view = cohort.loc[
        cohort["criterion"] == "lowest_RMSE",
        ["model", "mice", "percentage"],
    ]
    print(rmse_view.to_string(index=False))

    print(f"\n[{policy}] AICc winners (model-parameter k convention):")
    aicc_view = cohort.loc[
        cohort["criterion"] == "lowest_AICc_model_params_convention",
        ["model", "mice", "percentage"],
    ]
    print(aicc_view.to_string(index=False))

    focus = mouse_summary.loc[mouse_summary["ID"] == focus_id]
    if not focus.empty:
        print(f"\n[{policy}] Focus mouse {focus_id}:")
        cols = [
            "ID",
            "n",
            "exponential_RMSE",
            "logistic_RMSE",
            "gompertz_RMSE",
            "exponential_R2",
            "logistic_R2",
            "gompertz_R2",
            "best_RMSE_model",
        ]
        cols = [c for c in cols if c in focus.columns]
        print(focus[cols].to_string(index=False))

    return {
        "policy": policy,
        "results": results,
        "predictions": predictions,
        "mouse_summary": mouse_summary,
        "cohort": cohort,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit exponential, logistic, and Gompertz tumor-growth models "
            "independently to every mouse in LM2-4LUC.txt."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="LM2-4LUC.txt",
        help="Path to the LM2-4LUC text dataset.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="tumor_model_output",
        help="Output directory.",
    )
    parser.add_argument(
        "--zero-policy",
        choices=["retain", "exclude", "both"],
        default="both",
        help=(
            "How to handle observations recorded as exactly zero. "
            "'both' runs a sensitivity analysis."
        ),
    )
    parser.add_argument(
        "--focus-id",
        type=int,
        default=53,
        help="Mouse ID for extra focus CSV files / terminal summary.",
    )
    parser.add_argument(
        "--only-id",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Analyse only the listed mouse IDs. Useful for testing, e.g. "
            "--only-id 53."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all PNG plot generation.",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=2500,
        help="Maximum function evaluations per optimizer start.",
    )
    parser.add_argument(
        "--k-flag-ratio",
        type=float,
        default=10.0,
        help=(
            "Flag K as far above observed data when "
            "K/max(observed) exceeds this value."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print("LM2-4LUC tumor-growth model analysis")
    print("=" * 39)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    raw_df = load_data(input_path)
    audit, counts, zero_rows = audit_data(raw_df)

    # Save raw audit before applying any analysis policy.
    (output_path / "raw_data_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    counts.to_csv(
        output_path / "raw_measurements_per_mouse.csv",
        index=False,
    )
    zero_rows.to_csv(
        output_path / "raw_zero_observations.csv",
        index=False,
    )

    print("\nRaw-data audit:")
    print(f"  Rows:                 {audit['rows']}")
    print(f"  Unique mice:          {audit['unique_mice']}")
    print(
        f"  ID range:              "
        f"{audit['min_id']} to {audit['max_id']}"
    )
    print(f"  Zero observations:    {audit['zero_observations']}")
    print(
        f"  Negative observations:{audit['negative_observations']:>5}"
    )
    print(
        f"  Duplicate ID/time:    "
        f"{audit['duplicate_ID_Time_pairs']}"
    )
    print(
        f"  Measurements/mouse:   "
        f"{audit['measurements_per_mouse_min']} to "
        f"{audit['measurements_per_mouse_max']}"
    )

    if audit["negative_observations"] > 0:
        print(
            "\nERROR: Negative observations exist. "
            "Analysis stopped; inspect the raw data.",
            file=sys.stderr,
        )
        return 2

    if args.zero_policy == "both":
        policies = ["retain", "exclude"]
    else:
        policies = [args.zero_policy]

    run_index = []
    for policy in policies:
        run = analyse_policy(
            raw_df=raw_df,
            policy=policy,
            base_output=output_path,
            only_ids=args.only_id,
            focus_id=args.focus_id,
            make_plots=not args.no_plots,
            max_nfev=args.max_nfev,
            k_flag_ratio=args.k_flag_ratio,
            audit=audit,
        )
        run_index.append({
            "policy": policy,
            "mice_analysed": int(len(run["mouse_summary"])),
            "output_folder": str(
                output_path / f"zero_policy_{policy}"
            ),
        })

    pd.DataFrame(run_index).to_csv(
        output_path / "run_index.csv",
        index=False,
    )

    print("\nDone.")
    print(f"Results written to: {output_path}")
    print(
        "\nFor the IA, start with the focus-mouse files and "
        "mouse_summary.csv; use the cohort files for the "
        "all-mice extension."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
