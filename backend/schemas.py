"""Typed API payloads."""
import math
from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

StationId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_\-]{1,32}$")]
Measurement = Annotated[float | None, Field(default=None, ge=-1e6, le=1e6)]


class ObservationIn(BaseModel):
    """One AWS observation. A missing parameter is allowed (it is flagged by QC, not rejected);
    malformed values (non-numeric, NaN/inf) are rejected with 422."""
    model_config = ConfigDict(extra="forbid")
    station_id: StationId
    timestamp: datetime = Field(description="observation time (ISO 8601; naive = UTC)")
    temperature_c: Measurement = None
    pressure_hpa: Measurement = None
    relative_humidity_pct: Measurement = None
    source: Annotated[str | None, Field(max_length=64)] = None
    edge_flags: Annotated[list[Annotated[str, Field(max_length=32)]] | None, Field(max_length=16)] = None
    edge_health_score: Annotated[float | None, Field(ge=0, le=100)] = None

    @field_validator("temperature_c", "pressure_hpa", "relative_humidity_pct")
    @classmethod
    def finite(cls, v):
        if v is not None and not math.isfinite(v):
            raise ValueError("must be a finite number or null")
        return v

    @field_validator("timestamp")
    @classmethod
    def utc(cls, v: datetime):
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)


class BatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: Annotated[list[ObservationIn], Field(min_length=1, max_length=5000)]


class ObservationResult(BaseModel):
    station_id: str
    observation_time: str
    ingestion_time: str
    status: str
    reason: str | None = None
    temperature_c: float | None = None
    pressure_hpa: float | None = None
    relative_humidity_pct: float | None = None
    delayed: bool | None = None
    lag_minutes: float | None = None
    lag_reference: str | None = None
    out_of_order: bool | None = None
    detector_mode: str | None = None
    is_anomaly: bool | None = None
    anomaly_probability: float | None = Field(None, description="isotonic-calibrated P(anomaly); null in qc_only_fallback")
    anomaly_score: float | None = Field(None, description="uncalibrated detector output (not a probability)")
    fault_pattern: str | None = None
    parameter: str | None = None
    severity: str | None = None
    severity_score: float | None = None
    regional_evidence: bool | None = None
    regional_suppressed: bool | None = None
    weather_status: str | None = Field(None, description="NORMAL | REGIONAL_WEATHER | REGIONAL_WEATHER_WITH_STATION_EXCEPTION | SENSOR_ANOMALY | UNCERTAIN")
    regional_score: float | None = None
    health_score: float | None = None
    health_status: str | None = None
    estimated_temperature_c: float | None = None
    estimated_pressure_hpa: float | None = None
    estimated_relative_humidity_pct: float | None = None
    replacement_recommended: bool | None = None
    explanation: dict[str, Any] | None = None
    gap_event: dict[str, Any] | None = None
    processing_latency_ms: float | None = Field(None, description="engine time per observation (amortised over batch_size)")
    batch_size: int | None = None
    edge_flags: list[str] | None = None
    edge_health_score: float | None = None


class IngestResponse(BaseModel):
    accepted: int
    duplicates: int
    rejected: int
    results: list[ObservationResult]


class StationOut(BaseModel):
    station_id: str
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    source: str | None = None
    status: str
    label: str
    online: bool
    last_observation_time: str | None = None
    health_score: float | None = None
    health_status: str | None = None
    health_reasons: list[str] = []
    maintenance: dict[str, Any] = {}
    neighbors: list[dict[str, Any]] = []
    active_anomalies: int = 0
    active_critical: int = 0


class AnomalyOut(BaseModel):
    id: int
    observation_id: int | None
    station_id: str
    observation_time: str
    detected_at: str
    parameter: str | None
    fault_pattern: str
    severity: str | None
    anomaly_probability: float | None
    anomaly_score: float | None
    raw_value: float | None
    estimated_value: float | None
    regional_evidence: bool
    explanation: dict[str, Any] | None
    state: str
    detector_mode: str | None


class AnomalyUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Annotated[str, Field(pattern="^(OPEN|ACKNOWLEDGED|FALSE_ALARM|RESOLVED)$")]


class ReplayStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: Annotated[str, Field(pattern=r"^[a-z0-9_]{1,64}$")] = "sih_demo"
    speed: Annotated[float, Field(gt=0, le=1e7)] = 7200.0
    reset: bool = True


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    detector_mode: str
    model_version: str | None
    database_ok: bool
    stations: int
    auth_required_for_writes: bool
