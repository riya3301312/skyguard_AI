import tomllib
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@cache
def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else ROOT / "config" / "skyguard.toml"
    return tomllib.loads(p.read_text())
