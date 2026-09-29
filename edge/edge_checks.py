"""Python binding (ctypes) to the SAME C edge library that targets the ESP32.

Build it first: python edge/build_edge.py
"""
import ctypes
import math
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parent / "build" / ("skyguard_edge.dll" if sys.platform == "win32" else "libskyguard_edge.so")
VARS = ["temperature_c", "pressure_hpa", "relative_humidity_pct"]
KIND = ["MISSING", "RANGE", "SPIKE", "FLATLINE"]
SHORT = ["T", "P", "RH"]


class _Cfg(ctypes.Structure):
    _fields_ = [("min", ctypes.c_float * 3), ("max", ctypes.c_float * 3), ("max_step", ctypes.c_float * 3),
                ("flat_n", ctypes.c_uint16), ("mv_dt_min", ctypes.c_float), ("mv_drh_max", ctypes.c_float),
                ("mv_dtd_max", ctypes.c_float)]


class _State(ctypes.Structure):
    _fields_ = [("last", ctypes.c_float * 3), ("flat_run", ctypes.c_uint16 * 3), ("history", ctypes.c_uint32),
                ("has_last", ctypes.c_uint8)]


def decode(mask: int) -> list[str]:
    out = [f"{SHORT[v]}_{KIND[k]}" for v in range(3) for k in range(4) if mask & (1 << (4 * v + k))]
    if mask & (1 << 12):
        out.append("T_RH_INCONSISTENT")
    return out


class EdgeChecker:
    """Stateful per-station edge checker. Raises FileNotFoundError if the C library was not built."""

    def __init__(self, lib_path: Path = LIB):
        if not Path(lib_path).exists():
            raise FileNotFoundError(f"edge library not built: {lib_path} (run python edge/build_edge.py)")
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.sg_edge_check.restype = ctypes.c_uint32
        self.lib.sg_edge_state_size.restype = ctypes.c_size_t
        assert self.lib.sg_edge_state_size() == ctypes.sizeof(_State), "C/Python state layout mismatch"
        self.cfg, self.state = _Cfg(), _State()
        self.lib.sg_edge_default_config(ctypes.byref(self.cfg))
        self.lib.sg_edge_init(ctypes.byref(self.state))

    def check(self, obs: dict) -> dict:
        vals = [obs.get(v) for v in VARS]
        present = (ctypes.c_uint8 * 3)(*[int(x is not None and math.isfinite(float(x))) for x in vals])
        x = (ctypes.c_float * 3)(*[float(v) if p else 0.0 for v, p in zip(vals, present)])
        h = ctypes.c_uint8()
        mask = self.lib.sg_edge_check(ctypes.byref(self.state), ctypes.byref(self.cfg), x, present, ctypes.byref(h))
        return {"edge_flags": decode(mask), "edge_health_score": float(h.value)}
