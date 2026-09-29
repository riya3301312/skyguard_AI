"""Canonical SkyGuard observation schema."""
import pandas as pd

VARS = ["temperature_c", "pressure_hpa", "relative_humidity_pct"]
SHORT = {"temperature_c": "t", "pressure_hpa": "p", "relative_humidity_pct": "rh"}
LABEL = {"temperature_c": "Temperature", "pressure_hpa": "Pressure", "relative_humidity_pct": "Humidity"}
UNIT = {"temperature_c": "°C", "pressure_hpa": "hPa", "relative_humidity_pct": "%"}

CANONICAL = ["timestamp", "station_id", *VARS, "latitude", "longitude", "altitude_m", "source", "quality_flag"]
INJECTION = ["is_anomaly", "anomaly_type", "severity", "injected_parameter", "ground_truth_value", "injection_id"]
CORRECTION = ["estimated_temperature_c", "estimated_pressure_hpa", "estimated_relative_humidity_pct",
              "replacement_recommended"]

FAULT_CLASSES = ["NORMAL", "SPIKE", "NEGATIVE_SPIKE", "FLATLINE", "DRIFT", "STEP_OFFSET",
                 "MISSING_COMMUNICATION", "SENTINEL_CORRUPTION", "MULTIVARIATE_INCONSISTENCY",
                 "SPATIAL_INCONSISTENCY", "COMPOSITE_FAULT", "UNKNOWN_SUSPECT"]


def read_canonical(path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=True, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in ("injection_id", "injection_kind", "injected_parameter", "severity", "anomaly_type", "split_role"):
        if c in df:
            df[c] = df[c].fillna("").astype(str)
    return validate_canonical(df)


def validate_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError if df does not follow the canonical schema; return it unchanged otherwise."""
    missing = [c for c in CANONICAL if c not in df.columns]
    if missing:
        raise ValueError(f"missing canonical columns: {missing}")
    ts = df["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts) or ts.dt.tz is None:
        raise ValueError("timestamp must be timezone-aware datetime (UTC)")
    for c in VARS + ["latitude", "longitude", "altitude_m"]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"{c} must be numeric")
    if df["station_id"].isna().any():
        raise ValueError("station_id must not be null")
    if df.duplicated(["station_id", "timestamp"]).any():
        raise ValueError("duplicate (station_id, timestamp) rows")
    return df
