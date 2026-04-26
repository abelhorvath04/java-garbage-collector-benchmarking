#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WARMUP_CUTOFF_NS = 5 * 60 * 1_000_000_000
NS_PER_MS = 1_000_000
ROTATED_SUFFIXES = ["", ".0", ".1", ".2", ".3", ".4"]

ENV_DIRS = [
    ("G1GC-Java-25", "../experiment/G1GC-java25-2026-04-24"),
    ("G1GC-Java-21", "../experiment/G1GC-java21-2026-04-25"),
    ("ZGC-Java-25", "../experiment/ZGC-java25-2026-04-24"),
    ("ZGC-Java-21", "../experiment/ZGC-java21-2026-04-25"),
]

BENCHMARKS = [
    "akka-uct", "als", "chi-square", "dec-tree", "gauss-mix",
    "log-regression", "movie-lens", "naive-bayes", "page-rank",
    "fj-kmeans", "reactors", "db-shootout", "neo4j-analytics",
    "future-genetic", "mnemonics", "par-mnemonics", "rx-scrabble",
    "scrabble", "dotty", "philosophers", "scala-doku", "scala-kmeans",
    "scala-stm-bench7", "finagle-chirper", "finagle-http",
]

UPTIME_RE = re.compile(r"\[(?P<uptime>\d+(?:\.\d+)?)s\]")
TAGS_RE = re.compile(r"\]\[[a-zA-Z]+\s*\]\[(?P<tags>[^\]]+)\]")

PAUSE_RE = re.compile(
    r"""
    GC\(\d+\)\s+
    (?:[a-zA-Z]:\s+)?
    (?P<category>Pause\s+.*?)
    \s+
    (?P<duration>\d+(?:\.\d+)?)
    (?P<unit>ns|us|µs|ms|s)
    \s*$
    """,
    re.VERBOSE,
)

ZGC_STRICT_STW_CATEGORIES = {
    "Pause Mark Start",
    "Pause Mark End",
    "Pause Relocate Start",
}

KNOWN_PAUSE_PREFIXES = [
    "Pause Relocate Start",
    "Pause Mark Start",
    "Pause Mark End",
    "Pause Young",
    "Pause Mixed",
    "Pause Full",
    "Pause Remark",
    "Pause Cleanup",
    "Pause Init Mark",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StwAggregate:
    processed_gc_log_files: list[str]
    missing_gc_log_files: list[str]
    processed_gc_log_file_count: int
    stw_pause_count: int
    total_stw_ns: int
    total_stw_ms: float
    pause_breakdown: dict[str, dict[str, int | float]]
    min_gc_uptime_s_after_warmup: float | None
    max_gc_uptime_s_after_warmup: float | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_decimal(value: str, what: str) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid {what}: {value!r}") from exc


def parse_duration_to_ns(value: str, unit: str) -> int:
    unit = "us" if unit == "µs" else unit
    factors = {
        "ns": Decimal(1),
        "us": Decimal(1_000),
        "ms": Decimal(1_000_000),
        "s": Decimal(1_000_000_000),
    }
    if unit not in factors:
        raise ValueError(f"Unsupported duration unit: {unit!r}")

    return int(
        (parse_decimal(value, "duration") * factors[unit])
        .to_integral_value(rounding=ROUND_HALF_UP)
    )


def parse_uptime_to_ns(value_s: str) -> int:
    return int(
        (parse_decimal(value_s, "uptime seconds") * Decimal(1_000_000_000))
        .to_integral_value(rounding=ROUND_HALF_UP)
    )


def ns_to_ms(ns: int | float) -> float:
    return float(ns) / NS_PER_MS


def normalize_pause_category(category: str) -> str:
    category = " ".join(category.split())

    for prefix in KNOWN_PAUSE_PREFIXES:
        if (
            category == prefix
            or category.startswith(prefix + " ")
            or category.startswith(prefix + "(")
        ):
            return prefix

    if "(" in category:
        category = category.split("(", 1)[0].strip()

    return category.strip()


def category_to_column_slug(category: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def compute_stw_overhead_percent(total_stw_ms: float, total_duration_ms: float | None) -> float | None:
    if total_duration_ms is None or total_duration_ms <= 0:
        return None
    return (total_stw_ms / total_duration_ms) * 100.0


def detect_csv_dialect(path: Path) -> csv.Dialect:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)

    try:
        return csv.Sniffer().sniff(sample, delimiters=";,")
    except csv.Error:
        return csv.excel


def load_throughput_durations(path: Path) -> dict[tuple[str, str], float]:
    if not path.is_file():
        raise FileNotFoundError(f"Throughput CSV not found: {path}")

    dialect = detect_csv_dialect(path)
    durations: dict[tuple[str, str], float] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, dialect)
        header = next(reader, None)

        if not header:
            raise RuntimeError(f"Throughput CSV is empty: {path}")

        def first_index(column_name: str) -> int:
            try:
                return header.index(column_name)
            except ValueError as exc:
                raise RuntimeError(
                    f"Throughput CSV must contain column {column_name!r}. "
                    f"Found header: {header}"
                ) from exc

        benchmark_idx = first_index("benchmark")
        environment_idx = first_index("environment")
        total_duration_idx = first_index("total_duration_ms")

        for lineno, row in enumerate(reader, start=2):
            if len(row) <= max(benchmark_idx, environment_idx, total_duration_idx):
                print(f"WARNING: malformed throughput CSV row skipped at line {lineno}: {row}", file=sys.stderr)
                continue

            benchmark = row[benchmark_idx].strip()
            environment = row[environment_idx].strip()
            total_duration_raw = row[total_duration_idx].strip()

            if not benchmark or not environment or not total_duration_raw:
                continue

            try:
                total_duration_ms = float(total_duration_raw)
            except ValueError:
                print(
                    f"WARNING: invalid total_duration_ms skipped at line {lineno}: {total_duration_raw!r}",
                    file=sys.stderr,
                )
                continue

            durations[(benchmark, environment)] = total_duration_ms

    return durations


def collect_gc_logs(env_path: Path, benchmark: str) -> tuple[list[Path], list[Path]]:
    benchmark_dir = env_path / benchmark
    expected = [benchmark_dir / f"{benchmark}.gc.log{suffix}" for suffix in ROTATED_SUFFIXES]
    existing = [path for path in expected if path.is_file()]
    missing = [path for path in expected if not path.is_file()]
    return existing, missing


def is_strict_stw_pause(tags_raw: str, category: str) -> bool:
    tags = [tag.strip() for tag in tags_raw.split(",")]
    normalized = normalize_pause_category(category)

    # G1 / general HotSpot summary pause line.
    if tags == ["gc"]:
        return True

    # ZGC strict STW pause phases.
    if tags == ["gc", "phases"] and normalized in ZGC_STRICT_STW_CATEGORIES:
        return True

    return False


# ---------------------------------------------------------------------------
# GC log parsing
# ---------------------------------------------------------------------------

def parse_gc_logs(paths: list[Path], missing_paths: list[Path], warmup_cutoff_ns: int) -> StwAggregate:
    breakdown_ns: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "total_ns": 0})
    total_count = 0
    total_ns = 0
    min_uptime_s: float | None = None
    max_uptime_s: float | None = None

    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")

                if "Pause " not in line or "GC(" not in line:
                    continue

                uptime_match = UPTIME_RE.search(line)
                tags_match = TAGS_RE.search(line)
                pause_match = PAUSE_RE.search(line)

                if not uptime_match or not tags_match or not pause_match:
                    continue

                category = normalize_pause_category(pause_match.group("category"))
                tags_raw = tags_match.group("tags")

                if not is_strict_stw_pause(tags_raw, category):
                    continue

                uptime_s = float(uptime_match.group("uptime"))
                uptime_ns = parse_uptime_to_ns(uptime_match.group("uptime"))

                if uptime_ns < warmup_cutoff_ns:
                    continue

                duration_ns = parse_duration_to_ns(
                    pause_match.group("duration"),
                    pause_match.group("unit"),
                )

                breakdown_ns[category]["count"] += 1
                breakdown_ns[category]["total_ns"] += duration_ns
                total_count += 1
                total_ns += duration_ns

                min_uptime_s = uptime_s if min_uptime_s is None else min(min_uptime_s, uptime_s)
                max_uptime_s = uptime_s if max_uptime_s is None else max(max_uptime_s, uptime_s)

    pause_breakdown = {
        category: {
            "count": data["count"],
            "total_ms": ns_to_ms(data["total_ns"]),
        }
        for category, data in sorted(breakdown_ns.items())
    }

    return StwAggregate(
        processed_gc_log_files=[str(path) for path in paths],
        missing_gc_log_files=[str(path) for path in missing_paths],
        processed_gc_log_file_count=len(paths),
        stw_pause_count=total_count,
        total_stw_ns=total_ns,
        total_stw_ms=ns_to_ms(total_ns),
        pause_breakdown=pause_breakdown,
        min_gc_uptime_s_after_warmup=min_uptime_s,
        max_gc_uptime_s_after_warmup=max_uptime_s,
    )


# ---------------------------------------------------------------------------
# Summary building
# ---------------------------------------------------------------------------

def build_stw_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    throughput_durations = load_throughput_durations(args.throughput_csv)

    for env_name, env_path_raw in args.env:
        env_path = Path(env_path_raw)

        for benchmark in args.benchmark:
            try:
                gc_logs, missing_gc_logs = collect_gc_logs(env_path, benchmark)
                stw = parse_gc_logs(gc_logs, missing_gc_logs, args.warmup_cutoff_ns)

                total_duration_ms = throughput_durations.get((benchmark, env_name))
                stw_overhead_percent = compute_stw_overhead_percent(
                    stw.total_stw_ms,
                    total_duration_ms,
                )

            except Exception as exc:
                errors.append(f"{env_name}/{benchmark}: {exc}")
                continue

            rows.append({
                "benchmark": benchmark,
                "environment": env_name,
                "processed_gc_log_file_count": stw.processed_gc_log_file_count,
                "processed_gc_log_files": stw.processed_gc_log_files,
                "missing_gc_log_files": stw.missing_gc_log_files,
                "stw_pause_count": stw.stw_pause_count,
                "total_stw_ms": stw.total_stw_ms,
                "total_duration_ms": total_duration_ms,
                "stw_overhead_percent": stw_overhead_percent,
                "pause_breakdown": stw.pause_breakdown,
                "min_gc_uptime_s_after_warmup": stw.min_gc_uptime_s_after_warmup,
                "max_gc_uptime_s_after_warmup": stw.max_gc_uptime_s_after_warmup,
            })

    rows.sort(key=lambda r: (r["benchmark"], r["environment"]))

    validation = {
        "environment_count": len(args.env),
        "benchmark_count": len(args.benchmark),
        "expected_benchmark_environment_rows": len(args.env) * len(args.benchmark),
        "stw_rows_written": len(rows),
        "processed_gc_log_files": sum(r["processed_gc_log_file_count"] for r in rows),
        "missing_gc_log_groups": sum(1 for r in rows if r["processed_gc_log_file_count"] == 0),
        "rows_with_missing_rotated_gc_files": sum(1 for r in rows if r["missing_gc_log_files"]),
        "rows_with_total_duration_ms": sum(1 for r in rows if r["total_duration_ms"] is not None),
        "rows_without_total_duration_ms": sum(1 for r in rows if r["total_duration_ms"] is None),
        "total_strict_stw_pause_events_after_warmup": sum(r["stw_pause_count"] for r in rows),
        "total_stw_ms_after_warmup": sum(r["total_stw_ms"] for r in rows),
        "errors": errors,
    }

    return rows, validation


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def all_pause_categories(rows: list[dict[str, Any]]) -> list[str]:
    categories = set()
    for row in rows:
        categories.update(row["pause_breakdown"].keys())
    return sorted(categories)


def write_stw_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    categories = all_pause_categories(rows)

    category_fields: list[str] = []
    for category in categories:
        slug = category_to_column_slug(category)
        category_fields.extend([f"category_{slug}_count", f"category_{slug}_ms"])

    fields = [
        "benchmark",
        "environment",
        "processed_gc_log_file_count",
        "stw_pause_count",
        "total_stw_ms",
        "total_duration_ms",
        "stw_overhead_percent",
        "min_gc_uptime_s_after_warmup",
        "max_gc_uptime_s_after_warmup",
    ] + category_fields

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()

        for row in rows:
            out: dict[str, Any] = {
                "benchmark": row["benchmark"],
                "environment": row["environment"],
                "processed_gc_log_file_count": row["processed_gc_log_file_count"],
                "stw_pause_count": row["stw_pause_count"],
                "total_stw_ms": f"{row['total_stw_ms']:.6f}",
                "total_duration_ms": (
                    "" if row["total_duration_ms"] is None
                    else f"{row['total_duration_ms']:.6f}"
                ),
                "stw_overhead_percent": (
                    "" if row["stw_overhead_percent"] is None
                    else f"{row['stw_overhead_percent']:.6f}"
                ),
                "min_gc_uptime_s_after_warmup": (
                    "" if row["min_gc_uptime_s_after_warmup"] is None
                    else f"{row['min_gc_uptime_s_after_warmup']:.6f}"
                ),
                "max_gc_uptime_s_after_warmup": (
                    "" if row["max_gc_uptime_s_after_warmup"] is None
                    else f"{row['max_gc_uptime_s_after_warmup']:.6f}"
                ),
            }

            for category in categories:
                slug = category_to_column_slug(category)
                bucket = row["pause_breakdown"].get(category, {"count": 0, "total_ms": 0.0})
                out[f"category_{slug}_count"] = bucket["count"]
                out[f"category_{slug}_ms"] = f"{float(bucket['total_ms']):.6f}"

            writer.writerow(out)


def write_summary_json(
    path: Path,
    rows: list[dict[str, Any]],
    warmup_cutoff_ns: int,
    throughput_csv: Path,
    validation: dict[str, Any],
) -> None:
    payload = {
        "warmup_cutoff_ns": warmup_cutoff_ns,
        "throughput_csv": str(throughput_csv),
        "stw_pause_summary": rows,
        "validation": validation,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_env_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--env must be NAME=PATH")

    name, path = value.split("=", 1)

    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--env must be NAME=PATH")

    return name.strip(), path.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate post-warmup strict JVM STW pause times from rotated unified GC logs "
            "and compute STW overhead using post-warmup benchmark duration."
        )
    )

    parser.add_argument(
        "--env",
        action="append",
        type=parse_env_arg,
        default=None,
        help="Environment mapping NAME=PATH. Repeat for each environment. Overrides ENV_DIRS.",
    )

    parser.add_argument(
        "--benchmark",
        action="append",
        default=None,
        help="Benchmark name. Repeat to override the default 25 benchmark list.",
    )

    parser.add_argument(
        "--warmup-cutoff-ns",
        type=int,
        default=WARMUP_CUTOFF_NS,
        help="Warmup cutoff in nanoseconds. Default: 5 minutes.",
    )

    parser.add_argument(
        "--throughput-csv",
        type=Path,
        default=Path("throughput_summary_after_warmup.csv"),
        help="CSV containing benchmark/environment total_duration_ms after warmup.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output CSV/JSON files.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    args.env = args.env or ENV_DIRS
    args.benchmark = args.benchmark or BENCHMARKS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stw_rows, validation = build_stw_rows(args)

    csv_path = args.output_dir / "stw_pause_summary_after_warmup.csv"
    json_path = args.output_dir / "stw_pause_summary_after_warmup.json"

    write_stw_csv(csv_path, stw_rows)
    write_summary_json(
        json_path,
        stw_rows,
        args.warmup_cutoff_ns,
        args.throughput_csv,
        validation,
    )

    print("Validation summary")
    print(f"  environments processed: {validation['environment_count']}")
    print(f"  benchmarks processed: {validation['benchmark_count']}")
    print(f"  expected benchmark/environment rows: {validation['expected_benchmark_environment_rows']}")
    print(f"  STW rows written: {validation['stw_rows_written']}")
    print(f"  processed GC log files: {validation['processed_gc_log_files']}")
    print(f"  missing GC log groups: {validation['missing_gc_log_groups']}")
    print(f"  rows with missing rotated GC files: {validation['rows_with_missing_rotated_gc_files']}")
    print(f"  rows with total duration: {validation['rows_with_total_duration_ms']}")
    print(f"  rows without total duration: {validation['rows_without_total_duration_ms']}")
    print(f"  total strict STW pause events after warmup: {validation['total_strict_stw_pause_events_after_warmup']}")
    print(f"  total STW ms after warmup: {validation['total_stw_ms_after_warmup']:.6f}")
    print(f"  throughput CSV path: {args.throughput_csv}")
    print(f"  STW CSV path: {csv_path}")
    print(f"  STW JSON path: {json_path}")

    if validation["errors"]:
        print("\nErrors / skipped rows:", file=sys.stderr)
        for error in validation["errors"]:
            print(f"  - {error}", file=sys.stderr)
        return 1

    if validation["rows_without_total_duration_ms"]:
        print(
            "\nWARNING: Some rows have no matching total_duration_ms in the throughput CSV. "
            "For those rows, stw_overhead_percent is empty.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())