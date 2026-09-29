"""Build the edge C library and measure what is measurable without ESP32 hardware.

Produces:
  edge/build/skyguard_edge.(dll|so)  host shared library (used by the station simulator via ctypes)
  edge/build/edge_bench(.exe)        host benchmark harness
  edge/build/skyguard_edge_rv32.o    cross-compiled object for RV32IMC (ESP32-C3 core ISA)
  artifacts/edge/edge_metrics.json   measured sizes / host timings (no energy numbers: not measured)

Compiler: zig cc (pip package `ziglang`), i.e. clang/LLVM.
"""
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "edge" / "firmware"
BUILD = ROOT / "edge" / "build"
OUT = ROOT / "artifacts" / "edge"
ZIG = [sys.executable, "-m", "ziglang", "cc"]
RV = ["-target", "riscv32-linux-musl", "-mcpu=generic_rv32+m+c", "-Os", "-ffunction-sections", "-fdata-sections"]


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, capture_output=True, text=True, **kw)


def elf32_sections(path: Path) -> dict:
    """Sum allocated section sizes of an ELF32 little-endian relocatable object."""
    b = path.read_bytes()
    assert b[:4] == b"\x7fELF" and b[4] == 1, "expected ELF32"
    e_shoff, = struct.unpack_from("<I", b, 0x20)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", b, 0x2E)
    hdrs = [struct.unpack_from("<IIIIIIIIII", b, e_shoff + i * e_shentsize) for i in range(e_shnum)]
    strtab_off = hdrs[e_shstrndx][4]
    out = {"text": 0, "rodata": 0, "data": 0, "bss": 0, "sections": {}}
    for name_off, typ, flags, _, _, size, *_ in hdrs:
        name = b[strtab_off + name_off: b.index(b"\0", strtab_off + name_off)].decode()
        if not flags & 0x2:  # SHF_ALLOC
            continue
        out["sections"][name] = size
        if flags & 0x4:
            out["text"] += size
        elif typ == 8:
            out["bss"] += size
        elif flags & 0x1:
            out["data"] += size
        else:
            out["rodata"] += size
    return out


def stack_frames(asm: str) -> dict:
    frames, fn = {}, None
    for line in asm.splitlines():
        m = re.match(r"^([A-Za-z_]\w*):", line)
        if m:
            fn = m.group(1)
        m = re.search(r"addi\s+sp,\s*sp,\s*-(\d+)", line)
        if m and fn and fn not in frames:
            frames[fn] = int(m.group(1))
    return frames


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    src = str(FW / "skyguard_edge.c")
    lib = BUILD / ("skyguard_edge.dll" if sys.platform == "win32" else "libskyguard_edge.so")
    exe = BUILD / ("edge_bench.exe" if sys.platform == "win32" else "edge_bench")
    run(ZIG + ["-O2", "-shared", "-fPIC", "-o", str(lib), src])
    run(ZIG + ["-O2", "-o", str(exe), str(FW / "bench_host.c"), src])
    obj = BUILD / "skyguard_edge_rv32.o"
    run(ZIG + RV + ["-c", "-o", str(obj), src])
    asm_path = BUILD / "skyguard_edge_rv32.s"
    run(ZIG + RV + ["-S", "-o", str(asm_path), src])
    asm = asm_path.read_text()
    sizes = elf32_sections(obj)
    frames = stack_frames(asm)

    # host benchmark on real observations from the processed dataset (first station, first 20k rows)
    clean = ROOT / "data" / "processed" / "clean.csv"
    if clean.exists():
        df = pd.read_csv(clean, nrows=20000)
        lines = df[["temperature_c", "pressure_hpa", "relative_humidity_pct"]].to_csv(index=False, header=False, na_rep="")
    else:  # deterministic synthetic stream if the dataset was not built
        rng = np.random.default_rng(0)
        lines = "".join(f"{25 + rng.normal():.1f},{1008 + rng.normal():.1f},{60 + rng.normal():.1f}\n" for _ in range(20000))
    r = run([str(exe)], input=lines)
    bench = json.loads(r.stderr.strip().splitlines()[-1])
    flags = np.array([int(x.split(",")[0]) for x in r.stdout.split()])
    metrics = {
        "implementation": "Portable C99 library edge/firmware/skyguard_edge.c (same source as the ESP32 sketch)",
        "hardware_tested": False,
        "note": "No ESP32 hardware was available. Flash/RAM are measured from the RV32IMC (ESP32-C3 ISA) "
                "cross-compiled object; runtime is measured on the host CPU. Energy/power was NOT measured.",
        "flash_bytes_rv32imc_object": {"text": sizes["text"], "rodata": sizes["rodata"], "data": sizes["data"],
                                        "total": sizes["text"] + sizes["rodata"] + sizes["data"],
                                        "excludes": "libm logf and soft-float runtime helpers (resolved at link time on target)"},
        "ram_bytes": {"state_per_station": bench["state_bytes"], "config": bench["config_bytes"], "static_bss": sizes["bss"],
                      "stack_worst_case_rv32": frames.get("sg_edge_check", 0) + frames.get("dewpoint", 0),
                      "stack_frames_rv32": frames, "stack_note": "sg_edge_check -> dewpoint call chain; excludes libm logf"},
        "host_runtime": {"ns_per_observation": bench["ns_per_observation"], "observations": bench["observations"],
                         "repetitions": bench["repetitions"], "cpu": "host PC (x86-64), clang -O2"},
        "flag_rate_on_sample": float((flags > 0).mean()),
        "budget": {"flash_bytes": 8192, "ram_bytes": 1024, "host_ns_per_observation": 10000},
        "compiler": "zig cc (ziglang PyPI package, clang/LLVM)",
    }
    b = metrics["budget"]
    metrics["within_budget"] = bool(metrics["flash_bytes_rv32imc_object"]["total"] <= b["flash_bytes"]
                                    and bench["state_bytes"] + bench["config_bytes"] + metrics["ram_bytes"]["stack_worst_case_rv32"] <= b["ram_bytes"]
                                    and bench["ns_per_observation"] <= b["host_ns_per_observation"])
    (OUT / "edge_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("flash_bytes_rv32imc_object", "ram_bytes", "host_runtime", "within_budget")}, indent=2))


if __name__ == "__main__":
    main()
