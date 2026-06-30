#!/usr/bin/env python3
"""Run ChampSim configurations and export ROI metrics to CSV."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RunSpec:
    index: int
    total: int
    exe: Path
    trace: Path


@dataclass(frozen=True)
class RunResult:
    index: int
    row: dict[str, str]
    returncode: int
    timed_out: bool = False


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
    seen: set[Path] = set()
    for item in items:
        item_path = Path(item)
        search_pattern = str(item_path if item_path.is_absolute() else base / item_path)
        matches = sorted(Path(p) for p in glob.glob(search_pattern))
        if not matches:
            matches = [item_path]
        for path in matches:
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
            if path not in seen:
                resolved.append(path)
                seen.add(path)
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


def make_command(spec: RunSpec, args: argparse.Namespace) -> list[str]:
    return [
        str(spec.exe),
        "--warmup-instructions",
        str(args.warmup_instructions),
        "--simulation-instructions",
        str(args.simulation_instructions),
        *args.sim_arg,
        str(spec.trace),
    ]


def log_path_for(spec: RunSpec, args: argparse.Namespace) -> Path | None:
    if not args.log_dir:
        return None
    return args.log_dir / f"{spec.index:04d}_{spec.exe.name}_{slug_trace(spec.trace)}.log"


def run_buffered(command: list[str], args: argparse.Namespace) -> tuple[int, str, bool]:
    try:
        completed = subprocess.run(
            command,
            cwd=args.champsim_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        return completed.returncode, completed.stdout, False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return -9, f"{stdout}{stderr}\n[TIMEOUT]\n", True


def run_streaming(command: list[str], args: argparse.Namespace) -> tuple[int, str, bool]:
    output_parts: list[str] = []
    with subprocess.Popen(
        command,
        cwd=args.champsim_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            output_parts.append(line)
            print(line, end="")
        return proc.wait(), "".join(output_parts), False


def run_one(spec: RunSpec, args: argparse.Namespace, stream_output: bool) -> RunResult:
    command = make_command(spec, args)
    log_path = log_path_for(spec, args)

    if stream_output and args.timeout is None:
        returncode, output, timed_out = run_streaming(command, args)
    else:
        returncode, output, timed_out = run_buffered(command, args)

    if log_path:
        log_path.write_text(output, encoding="utf-8")

    row = parse_champsim_output(output)
    row.update(
        {
            "executable": spec.exe.name,
            "trace": slug_trace(spec.trace),
            "warmup_instructions": str(args.warmup_instructions),
            "simulation_instructions": str(args.simulation_instructions),
        }
    )

    if returncode != 0:
        reason = "timed out" if timed_out else f"exited with status {returncode}"
        print(f"warning: run {spec.index} {reason}", file=sys.stderr)

    return RunResult(spec.index, row, returncode, timed_out)


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
    parser.add_argument("--exe", action="append", required=True, help="ChampSim executable. Relative paths are resolved from --champsim-root.")
    parser.add_argument("--trace", action="append", required=True, help="Trace path or glob. Relative paths are resolved from --champsim-root.")
    parser.add_argument("--warmup-instructions", type=int, default=5_000_000)
    parser.add_argument("--simulation-instructions", type=int, default=20_000_000)
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Optional directory for raw ChampSim logs.")
    parser.add_argument("--append", action="store_true", help="Append new rows to an existing CSV.")
    parser.add_argument("--quiet", action="store_true", help="Do not print ChampSim stdout. Recommended with --jobs > 1.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of ChampSim processes to run concurrently.")
    parser.add_argument("--timeout", type=float, default=None, help="Per-run timeout in seconds.")
    parser.add_argument("--sim-arg", action="append", default=[], help="Extra argument passed to ChampSim before the trace path. Repeat if needed.")
    return parser.parse_args()


def print_run_start(spec: RunSpec, args: argparse.Namespace) -> None:
    command = make_command(spec, args)
    print(f"[{spec.index}/{spec.total}] {' '.join(shlex.quote(part) for part in command)}", flush=True)


def print_run_finish(result: RunResult, spec: RunSpec) -> None:
    status = "timeout" if result.timed_out else ("ok" if result.returncode == 0 else f"exit {result.returncode}")
    ipc = result.row.get("ipc", "")
    ipc_suffix = f" IPC={ipc}" if ipc else ""
    print(f"[{spec.index}/{spec.total}] done {status}: {spec.exe.name} {slug_trace(spec.trace)}{ipc_suffix}", flush=True)


def run_all(specs: list[RunSpec], args: argparse.Namespace) -> list[RunResult]:
    if args.jobs == 1:
        results = []
        for spec in specs:
            print_run_start(spec, args)
            result = run_one(spec, args, stream_output=not args.quiet)
            results.append(result)
            if args.quiet:
                print_run_finish(result, spec)
        return results

    print(f"Running {len(specs)} runs with {args.jobs} concurrent jobs", flush=True)
    results_by_index: dict[int, RunResult] = {}
    spec_by_index = {spec.index: spec for spec in specs}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_spec = {}
        for spec in specs:
            print_run_start(spec, args)
            future_to_spec[executor.submit(run_one, spec, args, False)] = spec

        try:
            for future in concurrent.futures.as_completed(future_to_spec):
                spec = future_to_spec[future]
                result = future.result()
                results_by_index[result.index] = result
                print_run_finish(result, spec)
        except KeyboardInterrupt:
            executor.shutdown(cancel_futures=True)
            raise

    return [results_by_index[index] for index in sorted(spec_by_index)]


def main() -> int:
    args = parse_args()
    args.champsim_root = args.champsim_root.resolve()
    if args.jobs < 1:
        print("error: --jobs must be at least 1", file=sys.stderr)
        return 2
    if args.log_dir:
        args.log_dir = args.log_dir.resolve()
        args.log_dir.mkdir(parents=True, exist_ok=True)

    executables = resolve_paths(args.exe, args.champsim_root)
    traces = resolve_paths(args.trace, args.champsim_root)

    missing = [path for path in [*executables, *traces] if not path.exists()]
    if missing:
        for path in missing:
            print(f"error: path does not exist: {path}", file=sys.stderr)
        return 2

    total = len(executables) * len(traces)
    specs = [
        RunSpec(index + 1, total, exe, trace)
        for index, (exe, trace) in enumerate((exe, trace) for exe in executables for trace in traces)
    ]

    results = run_all(specs, args)
    rows = [result.row for result in results]

    write_csv(args.output, rows, args.append)
    print(f"Wrote {len(rows)} new row(s) to {args.output}")
    return 1 if any(result.returncode != 0 for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
