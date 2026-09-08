"""
eval_common.py — shared config + evaluation for ALL models, for BOTH axes.

"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH  = Path("incidents_classified.csv")
OUTPUT_DIR = Path("model_outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

DATE_COL   = "date"

AXIS = "mechanism" 

_MECH_COL = "AI_Risk_Peril"
_LOB_COL  = "Line_of_Business"

MECH_CATEGORIES = [
    "Reliability",
    "Bias & Fairness",
    "Privacy, Confidentiality & Infringement",
    "Security & Misuse",
    "Autonomous Actions",
    "Governance, Oversight & Explainability",
]
LOB_CATEGORIES = [
    "Crime / Fidelity",
    "D&O",
    "Tech E&O",
    "Cyber",
    "EPLI",
]

def target_col(axis=None):
    return _MECH_COL if (axis or AXIS) == "mechanism" else _LOB_COL

def categories(axis=None):
    return MECH_CATEGORIES if (axis or AXIS) == "mechanism" else LOB_CATEGORIES

TARGET_COL = target_col()
MECH_ORDER = categories()   
REFERENCE_CATEGORY = MECH_ORDER[-1]
NAMED_MECHS = MECH_ORDER[:-1]   
MECH_KEEP   = MECH_ORDER[:-1] 

TEST_FOLDS = [(y, h) for y in range(2020, 2027) for h in (1, 2)
              if not (y == 2026 and h == 2)]  

HOLDOUT_FOLD = (2026, 2)
SERIES_START = "2016-01-01"
SERIES_END   = "2026-07-31"  

TEST_YEARS   = [2021, 2022, 2023, 2024, 2025]
HOLDOUT_YEAR = 2026


def set_axis(axis):
    """Flip the whole suite to 'mechanism' or 'lob'. Refreshes the convenience globals."""
    global AXIS, TARGET_COL, MECH_ORDER, REFERENCE_CATEGORY, NAMED_MECHS, MECH_KEEP
    assert axis in ("mechanism", "lob")
    AXIS = axis
    TARGET_COL = target_col()
    MECH_ORDER = categories()
    REFERENCE_CATEGORY = MECH_ORDER[-1]
    NAMED_MECHS = MECH_ORDER[:-1]
    MECH_KEEP   = MECH_ORDER[:-1]
    return AXIS


def fold_of(period):
    return (period.year, 1 if period.month <= 6 else 2)


def fold_label(year, half):
    return f"{year}-H{half}"


def fold_bounds(year, half):
    start_month = 1 if half == 1 else 7
    start = pd.Period(f"{year}-{start_month:02d}", freq="M")
    return start, start + 5      

def train_test_split_for(counts, year, half):
    test_start, test_end = fold_bounds(year, half)
    train = counts[counts.index < test_start]
    test  = counts[(counts.index >= test_start) & (counts.index <= test_end)]
    return train, test


def prev_fold(year, half):
    idx = TEST_FOLDS.index((year, half)) if (year, half) in TEST_FOLDS else None
    if idx is not None and idx > 0:
        return TEST_FOLDS[idx - 1]
    if half == 2:
        return (year, 1)
    return (year - 1, 2)


def load_incidents(path=None, axis=None, min_conf=None):
    axis = axis or AXIS
    col  = target_col(axis)
    cats = categories(axis)
    df = pd.read_csv(path or DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, col]).copy()
    if min_conf is not None and "Assignment_Confidence" in df.columns:
        df = df[df["Assignment_Confidence"] >= min_conf].copy()
    df = df[df[col].isin(cats)]
    df = df[(df[DATE_COL] >= SERIES_START) & (df[DATE_COL] <= SERIES_END)].copy()
    df["mechanism"] = df[col]
    df[target_col(axis)] = df[col]
    df["month_period"] = df[DATE_COL].dt.to_period("M")
    sort_cols = [DATE_COL] + (["incident_id"] if "incident_id" in df.columns else [])
    return df.sort_values(sort_cols).reset_index(drop=True)


def build_counts(data, axis=None):
    cats = categories(axis)
    col  = "mechanism" if "mechanism" in data.columns else target_col(axis)
    z = (data.groupby(["month_period", col], observed=True).size()
             .unstack(fill_value=0)
             .reindex(columns=cats, fill_value=0))
    z = z.reindex(pd.period_range(z.index.min(), z.index.max(), freq="M"), fill_value=0)
    z.index.name = "month_period"
    return z


def mlog(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1)
    p = p / p.sum()
    return np.sum(y * np.log(p)) / y.sum() if y.sum() > 0 else np.nan


def _period_col(x):
    for c in ("month_period", "quarter_period", "period"):
        if c in x.columns:
            return c
    raise KeyError("need a month_period column")


def _fold_col(x):
    if "fold" in x.columns:
        return "fold"
    if "test_year" in x.columns:
        return "test_year"
    raise KeyError("need a 'fold' column")


def evaluate(x):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    pc = _period_col(x)
    fc = _fold_col(x)
    ls = (x.groupby([fc, pc])
            .apply(lambda g: mlog(g.actual_count, g.predicted_share), include_groups=False))
    o = pd.DataFrame([{
        "model": x.model.iloc[0],
        "mae":  mean_absolute_error(x.actual_share, x.predicted_share),
        "rmse": mean_squared_error(x.actual_share, x.predicted_share) ** 0.5,
        "mean_log_score_per_incident": ls.mean(),
    }])
    y = (x.groupby(fc)
           .apply(lambda g: pd.Series({
               "mae":  mean_absolute_error(g.actual_share, g.predicted_share),
               "rmse": mean_squared_error(g.actual_share, g.predicted_share) ** 0.5,
           }), include_groups=False)
           .reset_index()
           .rename(columns={fc: "fold"}))
    y.insert(0, "model", x.model.iloc[0])
    return o, y