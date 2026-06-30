#!/usr/bin/env python3
"""Run ChampSim configurations and export key metrics to CSV."""

from __future__ import annotations

import argparse
import csv
import glob
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

BASE_COLUMNS = [
    "executable",
    "trace",
    "warmup_instructions",
    "simulation_instructions",
    "num_cpus",
    "page_size",
    "ipc",
    "instructions",
    "cycles",
    "branch_accuracy_pct",
    "branch_mpki",
    "avg_rob_occupancy_at_mispredict",
]


NUM_CPUS_RE = re.compile(r"^Number of CPUs: (?P<num_cpus>\d+)")
PAGE_SIZE_RE = re.compile(r"^Page size: (?P<page_size>\d+)")
CPU_RE = re.compile(r"^CPU (?P<cpu>\d+) cumulative IPC: (?P<ipc>\S+) instructions: (?P<instr>\d+) cycles: (?P<cycles>\d+)")
BRANCH_RE = re.compile(
    r"^CPU (?P<cpu>\d+) Branch Prediction Accuracy: (?P<acc>\S+)% MPKI: (?P<mpki>\S+) "
    r"Average ROB Occupancy at Mispredict: (?P<rob>\S+)"
)
BRANCH_TYPE_RE = re.compile(
    r"^(?P<kind>BRANCH_DIRECT_JUMP|BRANCH_INDIRECT|BRANCH_CONDITIONAL|BRANCH_DIRECT_CALL|"
    r"BRANCH_INDIRECT_CALL|BRANCH_RETURN|BRANCH_OTHER): (?P<mpki>\S+)"
)
CACHE_RE = re.compile(
    r"^cpu(?P<cpu>\d+)->(?P<cache>\S+)\s+(?P<kind>TOTAL|LOAD|RFO|PREFETCH|WRITE|TRANSLATION)\s+"
    r"ACCESS:\s+(?P<access>\d+)\s+HIT:\s+(?P<hit>\d+)\s+MISS:\s+(?P<miss>\d+)\s+"
    r"MSHR_MERGE:\s+(?P<mshr>\d+)"
)
PREFETCH_RE = re.compile(
    r"^cpu(?P<cpu>\d+)->(?P<cache>\S+) PREFETCH REQUESTED:\s+(?P<requested>\d+)\s+"
    r"ISSUED:\s+(?P<issued>\d+)\s+USEFUL:\s+(?P<useful>\d+)\s+USELESS:\s+(?P<useless>\d+)"
)
LATENCY_RE = re.compile(r"^cpu(?P<cpu>\d+)->(?P<cache>\S+) AVERAGE MISS LATENCY: (?P<latency>\S+) cycles")
DRAM_RQ_RE = re.compile(r"^(?P<channel>Channel \d+) RQ ROW_BUFFER_HIT:\s+(?P<hit>\d+)")
DRAM_WQ_RE = re.compile(r"^(?P<channel>Channel \d+) WQ ROW_BUFFER_HIT:\s+(?P<hit>\d+)")
DRAM_REFRESH_RE = re.compile(r"^(?P<channel>Channel \d+) REFRESHES ISSUED:\s+(?P<refresh>\d+)")
ROW_MISS_RE = re.compile(r"^\s+ROW_BUFFER_MISS:\s+(?P<miss>\d+)")
DBUS_RE = re.compile(r"^\s+AVG DBUS CONGESTED CYCLE:\s+(?P<avg>\S+)")
WQ_FULL_RE = re.compile(r"^\s+FULL:\s+(?P<full>\d+)")


def parse_number(value: str) -> str:
    return "" if value == "-" else value


def slug_trace(path: Path) -> str:
    name = path.name
    for suffix in (".champsimtrace.gz", ".champsimtrace.xz", ".champsimtrace", ".trace.gz", ".gz", ".xz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def resolve_paths(items: Iterable[str], base: Path) -> list[Path]:
    resolved: list[Path] = []
    for item in items:
        item_path = Path(item)
        search_pattern = str(item_path if item_path.is_absolute() else base / item_path)
        matches = [Path(p) for p in glob.glob(search_pattern)]
        if not matches:
            matches = [item_path]
        for path in matches:
            if not path.is_absolute():
                path = base / path
            resolved.append(path.resolve())
    return resolved


def parse_champsim_output(text: str) -> dict[str, str]:
    row: dict[str, str] = {}
    current_dram_channel = ""
    current_dram_queue = ""

    for line in text.splitlines():
        if match := NUM_CPUS_RE.match(line):
            row["num_cpus"] = match.group("num_cpus")
            continue

        if match := PAGE_SIZE_RE.match(line):
            row["page_size"] = match.group("page_size")
            continue

        if match := CPU_RE.match(line):
            row["ipc"] = parse_number(match.group("ipc"))
            row["instructions"] = match.group("instr")
            row["cycles"] = match.group("cycles")
            continue

        if match := BRANCH_RE.match(line):
            row["branch_accuracy_pct"] = parse_number(match.group("acc"))
            row["branch_mpki"] = parse_number(match.group("mpki"))
            row["avg_rob_occupancy_at_mispredict"] = parse_number(match.group("rob"))
            continue

        if match := BRANCH_TYPE_RE.match(line):
            row[f"{match.group('kind').lower()}_mpki"] = parse_number(match.group("mpki"))
            continue

        if match := CACHE_RE.match(line):
            cache = match.group("cache")
            kind = match.group("kind").lower()
            prefix = f"{cache}_{kind}"
            row[f"{prefix}_access"] = match.group("access")
            row[f"{prefix}_hit"] = match.group("hit")
            row[f"{prefix}_miss"] = match.group("miss")
            row[f"{prefix}_mshr_merge"] = match.group("mshr")
            continue

        if match := PREFETCH_RE.match(line):
            cache = match.group("cache")
            row[f"{cache}_pf_requested"] = match.group("requested")
            row[f"{cache}_pf_issued"] = match.group("issued")
            row[f"{cache}_pf_useful"] = match.group("useful")
            row[f"{cache}_pf_useless"] = match.group("useless")
            continue

        if match := LATENCY_RE.match(line):
            row[f"{match.group('cache')}_avg_miss_latency_cycles"] = parse_number(match.group("latency"))
            continue

        if match := DRAM_RQ_RE.match(line):
            current_dram_channel = match.group("channel").replace(" ", "_")
            current_dram_queue = "RQ"
            row[f"{current_dram_channel}_rq_row_buffer_hit"] = match.group("hit")
            continue

        if match := DRAM_WQ_RE.match(line):
            current_dram_channel = match.group("channel").replace(" ", "_")
            current_dram_queue = "WQ"
            row[f"{current_dram_channel}_wq_row_buffer_hit"] = match.group("hit")
            continue

        if match := ROW_MISS_RE.match(line):
            if current_dram_channel and current_dram_queue:
                queue = current_dram_queue.lower()
                row[f"{current_dram_channel}_{queue}_row_buffer_miss"] = match.group("miss")
            continue

        if match := DBUS_RE.match(line):
            if current_dram_channel:
                row[f"{current_dram_channel}_avg_dbus_congested_cycle"] = parse_number(match.group("avg"))
            continue

        if match := WQ_FULL_RE.match(line):
            if current_dram_channel:
                row[f"{current_dram_channel}_wq_full"] = match.group("full")
            continue

        if match := DRAM_REFRESH_RE.match(line):
            channel = match.group("channel").replace(" ", "_")
            row[f"{channel}_refreshes_issued"] = match.group("refresh")
            continue

    return row


def run_one(exe: Path, trace: Path, args: argparse.Namespace, run_id: int) -> dict[str, str]:
    log_path = None
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"{run_id:04d}_{exe.name}_{slug_trace(trace)}.log"

    command = [
        str(exe),
        "--warmup-instructions",
        str(args.warmup_instructions),
        "--simulation-instructions",
        str(args.simulation_instructions),
        *args.sim_arg,
        str(trace),
    ]

    print(f"[{run_id}] {' '.join(shlex.quote(part) for part in command)}", flush=True)
    output_parts: list[str] = []
    returncode = -1

    with subprocess.Popen(
        command,
        cwd=args.champsim_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        log_file = open(log_path, "w", encoding="utf-8") if log_path else None
        try:
            for line in proc.stdout:
                output_parts.append(line)
                if log_file:
                    log_file.write(line)
                if not args.quiet:
                    print(line, end="")
            returncode = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            output_parts.append("\n[TIMEOUT]\n")
            returncode = -9
        finally:
            if log_file:
                log_file.close()

    parsed = parse_champsim_output("".join(output_parts))
    if returncode != 0:
        print(f"warning: run {run_id} exited with status {returncode}", file=sys.stderr)

    parsed.update(
        {
            "executable": exe.name,
            "trace": slug_trace(trace),
            "warmup_instructions": str(args.warmup_instructions),
            "simulation_instructions": str(args.simulation_instructions),
        }
    )
    return parsed


def write_csv(path: Path, rows: list[dict[str, str]], append: bool) -> None:
    existing_rows: list[dict[str, str]] = []
    if append and path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    all_rows = [*existing_rows, *rows]
    dynamic_columns = sorted({key for row in all_rows for key in row} - set(BASE_COLUMNS))
    columns = [*BASE_COLUMNS, *dynamic_columns]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    default_champsim_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Run ChampSim and collect ROI metrics into CSV.")
    parser.add_argument("--champsim-root", type=Path, default=default_champsim_root, help="ChampSim repository path.")
    parser.add_argument("--exe", action="append", required=True, help="Executable path. Relative paths are resolved from --champsim-root.")
    parser.add_argument("--trace", action="append", required=True, help="Trace path or glob. Relative paths are resolved from --champsim-root.")
    parser.add_argument("--warmup-instructions", type=int, default=5_000_000)
    parser.add_argument("--simulation-instructions", type=int, default=20_000_000)
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Optional directory for raw ChampSim logs.")
    parser.add_argument("--append", action="store_true", help="Append to an existing CSV, rewriting it with a merged header.")
    parser.add_argument("--quiet", action="store_true", help="Do not stream ChampSim output to the terminal.")
    parser.add_argument("--timeout", type=float, default=None, help="Per-run timeout in seconds.")
    parser.add_argument("--sim-arg", action="append", default=[], help="Extra argument passed to ChampSim before the trace path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.champsim_root = args.champsim_root.resolve()
    if args.log_dir:
        args.log_dir = args.log_dir.resolve()

    executables = resolve_paths(args.exe, args.champsim_root)
    traces = resolve_paths(args.trace, args.champsim_root)

    missing = [path for path in [*executables, *traces] if not path.exists()]
    if missing:
        for path in missing:
            print(f"error: path does not exist: {path}", file=sys.stderr)
        return 2

    rows: list[dict[str, str]] = []
    run_id = 1
    for exe in executables:
        for trace in traces:
            rows.append(run_one(exe, trace, args, run_id))
            run_id += 1

    write_csv(args.output, rows, args.append)
    print(f"Wrote {len(rows)} new row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
