"""Download real Indian surface observations (IMD aerodrome METAR reports) from the
Iowa Environmental Mesonet (IEM) archive into data/raw/ exactly as received.

Raw bytes are stored gzip-compressed and never modified afterwards. A provenance
manifest (URL, retrieval time, SHA-256 of the uncompressed bytes) is written to
data/raw/iem_metar/PROVENANCE.json.

This is the ONLY step that needs internet. Everything downstream runs offline
from data/raw/.
"""
import argparse
import gzip
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.config import load_config  # noqa: E402

IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def fetch(station: str, start: str, end: str) -> tuple[str, bytes]:
    y1, m1, d1 = start.split("-")
    y2, m2, d2 = end.split("-")
    params = [("station", station)] + [("data", d) for d in ("tmpc", "relh", "alti", "mslp")] + [
        ("year1", y1), ("month1", m1), ("day1", d1), ("year2", y2), ("month2", m2), ("day2", d2),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "yes"), ("elev", "yes"),
        ("missing", "M"), ("trace", "T"), ("direct", "no"), ("report_type", "3"), ("report_type", "4"),
    ]
    url = f"{IEM}?{urlencode(params)}"
    for attempt in range(6):
        try:
            with urlopen(url, timeout=180) as r:
                return url, r.read()
        except (HTTPError, ConnectionError, TimeoutError) as e:
            if (isinstance(e, HTTPError) and e.code != 429) or attempt == 5:
                raise
            time.sleep(15 * (attempt + 1))  # IEM rate limit / transient reset


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=cfg["data"]["start"])
    ap.add_argument("--end", default=cfg["data"]["end"])
    ap.add_argument("--stations", nargs="*", default=[s["id"] for s in cfg["data"]["stations"]])
    args = ap.parse_args()

    out = ROOT / "data" / "raw" / "iem_metar"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "PROVENANCE.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    manifest.update({
        "source": "Iowa Environmental Mesonet (IEM) ASOS/METAR archive, network IN__ASOS",
        "original_producer": "India Meteorological Department (IMD) aerodrome meteorological offices "
                             "(METAR/SPECI reports exchanged via WMO GTS)",
        "api": IEM,
        "license_note": "METAR reports are freely exchanged WMO essential data; IEM redistributes them publicly.",
        "variables": {"tmpc": "air temperature degC (METAR resolution 1 degC)",
                      "relh": "relative humidity % (IEM-derived from reported T and dew point)",
                      "alti": "altimeter setting QNH, inHg (sea-level-reduced pressure)",
                      "mslp": "sea level pressure hPa (rarely reported for India, usually M)"},
        "caveat": "These are aerodrome surface observations, not the IMD AWS network feed. "
                  "They are used as a real Indian AWS-like observation stream.",
    })
    for st in args.stations:
        dest = out / f"{st}.csv.gz"
        if dest.exists():
            print(f"{st}: exists, skipping (raw data is never overwritten)")
            continue
        try:
            url, body = fetch(st, args.start, args.end)
        except OSError as e:  # includes HTTPError / connection resets after retries
            print(f"{st}: FAILED ({e}); skipped")
            manifest.setdefault("failed", {})[st] = str(e)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            continue
        if body.count(b"\n") < 2:
            print(f"{st}: no observations in archive; skipped")
            manifest.setdefault("empty", []).append(st)
            continue
        dest.write_bytes(gzip.compress(body, mtime=0))
        manifest["files"][dest.name] = {
            "station": st, "url": url, "start": args.start, "end": args.end,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256_uncompressed": hashlib.sha256(body).hexdigest(), "bytes": len(body),
        }
        print(f"{st}: {len(body)} bytes")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        time.sleep(10)  # be polite to IEM (it rate-limits bursts)


if __name__ == "__main__":
    main()
