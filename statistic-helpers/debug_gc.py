#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WARMUP_CUTOFF_NS = 5 * 60 * 1_000_000_000
NS_PER_MS = 1_000_000
ROTATED_SUFFIXES = ["", ".0", ".1", ".2", ".3", ".4"]

# Adjust these paths if your experiment folders move.
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

# ---------------------------------------------------------------------------
# GC pause parsing configuration
# ---------------------------------------------------------------------------

# Broad GC pause matcher.
#
# Purpose:
#   This regex should capture candidate pause lines without already deciding
#   whether they are valid STW events.
#
# Examples it should capture:
#
#   [2026-04-24T17:22:17.821+0000][592.082s][info ][gc                   ] GC(141) Pause Young ... 126.334ms
#   [2026-04-25T06:10:35.809+0000][610.945s][info ][gc,phases            ] GC(278) y: Pause Mark Start 0.030ms
#
BROAD_GC_PAUSE_RE = re.compile(
    r"""
    ^.*?
    \[(?P<uptime>\d+(?:\.\d+)?)s\]
    \[[^\]]+\]
    \[(?P<tags>[^\]]+)\]\s+
    GC\((?P<gc_id>\d+)\)\s+
    (?P<body>.*\bPause\b.*)
    $
    """,
    re.VERBOSE,
)

# Duration tokens inside the GC log body.
#
# The parser uses the first duration token in the body as the pause duration.
# This is intentional: if a JVM log line also appends User/Sys/Real times after
# the pause duration, using the last duration token would accidentally capture
# Real=... instead of the actual pause duration.
DURATION_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ns|us|µs|ms|s)\b"
)

ZGC_STRICT_STW_CATEGORIES = {
    "Pause Mark Start",
    "Pause Mark End",
    "Pause Relocate Start",
}

G1_STRICT_STW_CATEGORIES = {
    "Pause Young",
    "Pause Mixed",
    "Pause Full",
    "Pause Remark",
    "Pause Cleanup",
    "Pause Init Mark",
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
class BenchmarkDurations:
    measurement_count_after_warmup: int
    total_duration_ns: int
    total_duration_ms: float
    avg_duration_ms: float


@dataclass(frozen=True)
class StwAggregate:
    processed_gc_log_files: list[str]
    missing_gc_log_files: list[str]
    processed_gc_log_file_count: int

    stw_pause_count: int
    stw_nonzero_pause_count: int
    stw_zero_duration_pause_count: int
    stw_gc_id_count_with_stw: int

    total_stw_ns: int
    total_stw_ms: float

    pause_breakdown: dict[str, dict[str, int | float]]
    min_gc_uptime_s_after_warmup: float | None
    max_gc_uptime_s_after_warmup: float | None

    debug_events: list[dict[str, Any]]
    audit_counters: dict[str, int]


# ---------------------------------------------------------------------------
# Numeric helpers
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


def compute_overhead_percent(total_stw_ns: int, total_duration_ns: int) -> float:
    if total_duration_ns <= 0:
        return 0.0
    return (total_stw_ns / total_duration_ns) * 100.0


# ---------------------------------------------------------------------------
# GC log parsing helpers
# ---------------------------------------------------------------------------

def normalize_tags(tags_raw: str) -> list[str]:
    return [tag.strip() for tag in tags_raw.split(",") if tag.strip()]


def tags_to_string(tags_raw: str) -> str:
    return ",".join(normalize_tags(tags_raw))


def normalize_pause_category(category: str) -> str:
    category = " ".join(category.split())

    # ZGC phase lines may contain prefixes such as:
    #   y: Pause Mark Start 0.030ms
    #   m: Pause Mark End 0.012ms
    category = re.sub(r"^[A-Za-z0-9_.$-]+:\s+", "", category)

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


def collect_gc_logs(env_path: Path, benchmark: str) -> tuple[list[Path], list[Path]]:
    benchmark_dir = env_path / benchmark
    expected = [benchmark_dir / f"{benchmark}.gc.log{suffix}" for suffix in ROTATED_SUFFIXES]
    existing = [path for path in expected if path.is_file()]
    missing = [path for path in expected if not path.is_file()]
    return existing, missing


def extract_category_and_duration(body: str) -> tuple[str, str, int, float] | None:
    """
    Extract raw category, normalized category and duration.

    Important:
      We use the first duration-looking token in the body, not the last one.

    Reason:
      Some GC log formats may append CPU accounting such as User/Sys/Real after
      the pause duration. If we used the last duration token, we could accidentally
      parse Real=... instead of the pause duration.
    """
    matches = list(DURATION_RE.finditer(body))

    if not matches:
        return None

    duration_match = matches[0]

    duration_ns = parse_duration_to_ns(
        duration_match.group("value"),
        duration_match.group("unit"),
    )
    duration_ms = ns_to_ms(duration_ns)

    raw_category = body[:duration_match.start()].strip()
    raw_category = " ".join(raw_category.split())

    normalized_category = normalize_pause_category(raw_category)

    return raw_category, normalized_category, duration_ns, duration_ms


def classify_pause_candidate(
    *,
    tags_raw: str,
    normalized_category: str,
    uptime_ns: int,
    warmup_cutoff_ns: int,
    zgc_relaxed_tags: bool,
) -> tuple[bool, str]:
    """
    Return:
      accepted, status

    Accepted means:
      this candidate is included in stw_pause_count and total_stw_ms.

    Status is written to debug CSV so the decision is auditable.
    """
    tags = normalize_tags(tags_raw)

    if uptime_ns < warmup_cutoff_ns:
        return False, "rejected_before_warmup"

    # G1 / general unified GC summary pause lines.
    #
    # This accepts exact [gc] pause summary lines only.
    # It intentionally does not accept [gc,start], [gc,heap], [gc,cpu],
    # or [gc,phases] lines, because those are not the final pause summary.
    if tags == ["gc"] and normalized_category in G1_STRICT_STW_CATEGORIES:
        return True, "accepted_g1_strict_pause"

    # Strict ZGC mode:
    # accept only [gc,phases] with known short STW phase categories.
    if tags == ["gc", "phases"] and normalized_category in ZGC_STRICT_STW_CATEGORIES:
        return True, "accepted_zgc_strict_phase"

    # Optional relaxed ZGC mode:
    # accept strict ZGC STW categories if "gc" is present anywhere in the tag list.
    #
    # Use this only after inspecting debug output. It is useful if your JVM logs
    # ZGC phase lines with slightly different tag combinations.
    if (
        zgc_relaxed_tags
        and "gc" in tags
        and normalized_category in ZGC_STRICT_STW_CATEGORIES
    ):
        return True, "accepted_zgc_relaxed_tag"

    if "gc" in tags and normalized_category in ZGC_STRICT_STW_CATEGORIES:
        return False, "candidate_zgc_category_but_tag_not_strict"

    if normalized_category.startswith("Pause"):
        return False, "candidate_pause_but_category_or_tag_rejected"

    return False, "rejected_other"


def stable_line_hash(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8", errors="replace")).hexdigest()[:16]


def should_include_debug_event(
    *,
    args: argparse.Namespace,
    benchmark: str,
    environment: str,
) -> bool:
    if not args.debug_stw:
        return False

    if args.debug_benchmark and benchmark != args.debug_benchmark:
        return False

    if args.debug_environment and environment != args.debug_environment:
        return False

    return True


def parse_gc_logs(
    *,
    paths: list[Path],
    missing_paths: list[Path],
    warmup_cutoff_ns: int,
    benchmark: str,
    environment: str,
    args: argparse.Namespace,
) -> StwAggregate:
    breakdown_ns: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "total_ns": 0}
    )

    total_count = 0
    nonzero_count = 0
    zero_duration_count = 0
    total_ns = 0

    accepted_gc_ids: set[str] = set()
    accepted_event_keys: Counter[tuple[str, str, str, int, int]] = Counter()

    min_uptime_s: float | None = None
    max_uptime_s: float | None = None

    debug_events: list[dict[str, Any]] = []
    audit_counters: Counter[str] = Counter()

    include_debug = should_include_debug_event(
        args=args,
        benchmark=benchmark,
        environment=environment,
    )

    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.rstrip("\n")

                # Very cheap pre-filter.
                if "Pause" not in line or "GC(" not in line:
                    continue

                audit_counters["raw_pause_gc_candidates"] += 1

                match = BROAD_GC_PAUSE_RE.match(line)
                if not match:
                    audit_counters["broad_regex_not_matched"] += 1

                    if include_debug:
                        debug_events.append({
                            "benchmark": benchmark,
                            "environment": environment,
                            "gc_log_file": str(path),
                            "line_number": line_number,
                            "status": "broad_regex_not_matched",
                            "accepted": "FALSE",
                            "uptime_s": "",
                            "gc_id": "",
                            "tags": "",
                            "raw_category": "",
                            "normalized_category": "",
                            "duration_ns": "",
                            "duration_ms": "",
                            "line_hash": stable_line_hash(line),
                            "raw_line": line,
                        })

                    continue

                audit_counters["broad_regex_matched"] += 1

                tags_raw = match.group("tags")
                tags_clean = tags_to_string(tags_raw)
                uptime_s = float(match.group("uptime"))
                uptime_ns = parse_uptime_to_ns(match.group("uptime"))
                gc_id = match.group("gc_id")
                body = match.group("body")

                extracted = extract_category_and_duration(body)

                if extracted is None:
                    audit_counters["no_duration_token"] += 1

                    if include_debug:
                        debug_events.append({
                            "benchmark": benchmark,
                            "environment": environment,
                            "gc_log_file": str(path),
                            "line_number": line_number,
                            "status": "no_duration_token",
                            "accepted": "FALSE",
                            "uptime_s": f"{uptime_s:.6f}",
                            "gc_id": gc_id,
                            "tags": tags_clean,
                            "raw_category": body,
                            "normalized_category": normalize_pause_category(body),
                            "duration_ns": "",
                            "duration_ms": "",
                            "line_hash": stable_line_hash(line),
                            "raw_line": line,
                        })

                    continue

                raw_category, normalized_category, duration_ns, duration_ms = extracted

                accepted, status = classify_pause_candidate(
                    tags_raw=tags_raw,
                    normalized_category=normalized_category,
                    uptime_ns=uptime_ns,
                    warmup_cutoff_ns=warmup_cutoff_ns,
                    zgc_relaxed_tags=args.zgc_relaxed_tags,
                )

                audit_counters[status] += 1

                if duration_ns == 0:
                    audit_counters["candidate_zero_duration"] += 1
                else:
                    audit_counters["candidate_nonzero_duration"] += 1

                # Event key used for duplicate diagnostics.
                # This does not automatically remove events.
                event_key = (
                    gc_id,
                    tags_clean,
                    normalized_category,
                    uptime_ns,
                    duration_ns,
                )
                accepted_event_keys[event_key] += 1

                if accepted:
                    breakdown_ns[normalized_category]["count"] += 1
                    breakdown_ns[normalized_category]["total_ns"] += duration_ns

                    total_count += 1
                    total_ns += duration_ns
                    accepted_gc_ids.add(gc_id)

                    if duration_ns > 0:
                        nonzero_count += 1
                    else:
                        zero_duration_count += 1

                    min_uptime_s = uptime_s if min_uptime_s is None else min(min_uptime_s, uptime_s)
                    max_uptime_s = uptime_s if max_uptime_s is None else max(max_uptime_s, uptime_s)

                if include_debug:
                    duplicate_key_count = accepted_event_keys[event_key]

                    debug_events.append({
                        "benchmark": benchmark,
                        "environment": environment,
                        "gc_log_file": str(path),
                        "line_number": line_number,
                        "status": status,
                        "accepted": "TRUE" if accepted else "FALSE",
                        "uptime_s": f"{uptime_s:.6f}",
                        "gc_id": gc_id,
                        "tags": tags_clean,
                        "raw_category": raw_category,
                        "normalized_category": normalized_category,
                        "duration_ns": duration_ns,
                        "duration_ms": f"{duration_ms:.9f}",
                        "line_hash": stable_line_hash(line),
                        "duplicate_event_key_count_so_far": duplicate_key_count,
                        "raw_line": line,
                    })

    duplicate_accepted_event_keys = sum(
        1 for count in accepted_event_keys.values() if count > 1
    )
    audit_counters["duplicate_accepted_event_keys"] = duplicate_accepted_event_keys

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
        stw_nonzero_pause_count=nonzero_count,
        stw_zero_duration_pause_count=zero_duration_count,
        stw_gc_id_count_with_stw=len(accepted_gc_ids),

        total_stw_ns=total_ns,
        total_stw_ms=ns_to_ms(total_ns),

        pause_breakdown=pause_breakdown,
        min_gc_uptime_s_after_warmup=min_uptime_s,
        max_gc_uptime_s_after_warmup=max_uptime_s,

        debug_events=debug_events,
        audit_counters=dict(audit_counters),
    )


# ---------------------------------------------------------------------------
# Renaissance measurement parsing
# ---------------------------------------------------------------------------

def iter_json_measurements(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        if "duration_ns" in obj and "uptime_ns" in obj:
            yield obj
        for value in obj.values():
            yield from iter_json_measurements(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_json_measurements(item)


def parse_benchmark_json(path: Path, warmup_cutoff_ns: int) -> BenchmarkDurations:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    durations: list[int] = []
    malformed = 0

    for record in iter_json_measurements(data):
        try:
            uptime_ns = int(record["uptime_ns"])
            duration_ns = int(record["duration_ns"])
        except Exception:
            malformed += 1
            continue

        if uptime_ns >= warmup_cutoff_ns:
            durations.append(duration_ns)

    if malformed:
        print(
            f"WARNING: skipped {malformed} malformed JSON measurement records in {path}",
            file=sys.stderr,
        )

    if not durations:
        raise RuntimeError(f"No benchmark measurements left after warmup cutoff in {path}")

    total_ns = sum(durations)
    avg_ns = total_ns / len(durations)

    return BenchmarkDurations(
        measurement_count_after_warmup=len(durations),
        total_duration_ns=total_ns,
        total_duration_ms=ns_to_ms(total_ns),
        avg_duration_ms=ns_to_ms(avg_ns),
    )


def parse_benchmark_csv(path: Path, warmup_cutoff_ns: int) -> BenchmarkDurations:
    durations: list[int] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        if not {"duration_ns", "uptime_ns"}.issubset(fieldnames):
            raise RuntimeError(f"Missing duration_ns/uptime_ns columns in {path}")

        for lineno, row in enumerate(reader, start=2):
            try:
                uptime_ns = int(row["uptime_ns"])
                duration_ns = int(row["duration_ns"])
            except Exception as exc:
                raise RuntimeError(f"Malformed CSV row in {path}:{lineno}: {row}") from exc

            if uptime_ns >= warmup_cutoff_ns:
                durations.append(duration_ns)

    if not durations:
        raise RuntimeError(f"No benchmark measurements left after warmup cutoff in {path}")

    total_ns = sum(durations)
    avg_ns = total_ns / len(durations)

    return BenchmarkDurations(
        measurement_count_after_warmup=len(durations),
        total_duration_ns=total_ns,
        total_duration_ms=ns_to_ms(total_ns),
        avg_duration_ms=ns_to_ms(avg_ns),
    )


def find_measurement_file(env_path: Path, benchmark: str, mode: str) -> Path:
    benchmark_dir = env_path / benchmark

    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    preferred_json = benchmark_dir / f"{benchmark}.json"
    preferred_csv = benchmark_dir / f"{benchmark}.csv"

    if mode == "json":
        if preferred_json.exists():
            return preferred_json
        candidates = sorted(benchmark_dir.glob("*.json"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"No JSON measurement file found for {benchmark} under {benchmark_dir}")

    if mode == "csv":
        if preferred_csv.exists():
            return preferred_csv
        candidates = sorted(benchmark_dir.glob("*.csv"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"No CSV measurement file found for {benchmark} under {benchmark_dir}")

    # auto mode: prefer the canonical benchmark-named files, then any JSON, then any CSV.
    if preferred_json.exists():
        return preferred_json

    if preferred_csv.exists():
        return preferred_csv

    json_candidates = sorted(benchmark_dir.glob("*.json"))
    if json_candidates:
        return json_candidates[0]

    csv_candidates = sorted(benchmark_dir.glob("*.csv"))
    if csv_candidates:
        return csv_candidates[0]

    raise FileNotFoundError(f"No JSON or CSV measurement file found for {benchmark} under {benchmark_dir}")


def parse_benchmark_measurements(
    env_path: Path,
    benchmark: str,
    mode: str,
    warmup_cutoff_ns: int,
) -> tuple[BenchmarkDurations, str]:
    path = find_measurement_file(env_path, benchmark, mode)

    if path.suffix.lower() == ".json":
        return parse_benchmark_json(path, warmup_cutoff_ns), str(path)

    if path.suffix.lower() == ".csv":
        return parse_benchmark_csv(path, warmup_cutoff_ns), str(path)

    raise RuntimeError(f"Unsupported measurement file type: {path}")


# ---------------------------------------------------------------------------
# Summary building
# ---------------------------------------------------------------------------

def compute_fastest_by_benchmark(
    throughput_rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in throughput_rows:
        grouped[row["benchmark"]].append(row)

    fastest: dict[str, dict[str, Any]] = {}

    for benchmark, rows in grouped.items():
        # Lower average duration after warmup means faster.
        fastest[benchmark] = min(rows, key=lambda r: (r["avg_duration_ms"], r["environment"]))

    return fastest


def build_summaries(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    throughput_rows: list[dict[str, Any]] = []
    stw_rows: list[dict[str, Any]] = []
    debug_events: list[dict[str, Any]] = []
    errors: list[str] = []

    global_audit_counters: Counter[str] = Counter()

    for env_name, env_path_raw in args.env:
        env_path = Path(env_path_raw)

        for benchmark in args.benchmark:
            try:
                durations, measurement_file = parse_benchmark_measurements(
                    env_path=env_path,
                    benchmark=benchmark,
                    mode=args.measurements,
                    warmup_cutoff_ns=args.warmup_cutoff_ns,
                )

                gc_logs, missing_gc_logs = collect_gc_logs(env_path, benchmark)

                stw = parse_gc_logs(
                    paths=gc_logs,
                    missing_paths=missing_gc_logs,
                    warmup_cutoff_ns=args.warmup_cutoff_ns,
                    benchmark=benchmark,
                    environment=env_name,
                    args=args,
                )

            except Exception as exc:
                errors.append(f"{env_name}/{benchmark}: {exc}")
                continue

            throughput_rows.append({
                "benchmark": benchmark,
                "environment": env_name,
                "measurement_file": measurement_file,
                "measurement_count": durations.measurement_count_after_warmup,
                "total_duration_ms": durations.total_duration_ms,
                "avg_duration_ms": durations.avg_duration_ms,
                "_total_duration_ns": durations.total_duration_ns,
            })

            stw_rows.append({
                "benchmark": benchmark,
                "environment": env_name,
                "processed_gc_log_file_count": stw.processed_gc_log_file_count,
                "processed_gc_log_files": stw.processed_gc_log_files,
                "missing_gc_log_files": stw.missing_gc_log_files,

                "stw_pause_count": stw.stw_pause_count,
                "stw_nonzero_pause_count": stw.stw_nonzero_pause_count,
                "stw_zero_duration_pause_count": stw.stw_zero_duration_pause_count,
                "stw_gc_id_count_with_stw": stw.stw_gc_id_count_with_stw,

                "total_stw_ms": stw.total_stw_ms,
                "stw_overhead_percent": compute_overhead_percent(
                    stw.total_stw_ns,
                    durations.total_duration_ns,
                ),

                "pause_breakdown": stw.pause_breakdown,
                "min_gc_uptime_s_after_warmup": stw.min_gc_uptime_s_after_warmup,
                "max_gc_uptime_s_after_warmup": stw.max_gc_uptime_s_after_warmup,
                "audit_counters": stw.audit_counters,
            })

            debug_events.extend(stw.debug_events)
            global_audit_counters.update(stw.audit_counters)

    fastest = compute_fastest_by_benchmark(throughput_rows) if throughput_rows else {}

    for row in throughput_rows:
        winner = fastest[row["benchmark"]]
        is_fastest = row["environment"] == winner["environment"]

        row["fastest_environment"] = winner["environment"] if is_fastest else ""
        row["fastest_avg_duration_ms"] = winner["avg_duration_ms"] if is_fastest else ""
        row["is_fastest_environment"] = is_fastest

        del row["_total_duration_ns"]

    throughput_rows.sort(key=lambda r: (r["benchmark"], r["environment"]))
    stw_rows.sort(key=lambda r: (r["benchmark"], r["environment"]))

    validation = {
        "environment_count": len(args.env),
        "benchmark_count": len(args.benchmark),
        "expected_benchmark_environment_rows": len(args.env) * len(args.benchmark),

        "throughput_rows_written": len(throughput_rows),
        "stw_rows_written": len(stw_rows),

        "processed_gc_log_files": sum(r["processed_gc_log_file_count"] for r in stw_rows),
        "missing_gc_log_groups": sum(1 for r in stw_rows if r["processed_gc_log_file_count"] == 0),
        "rows_with_missing_rotated_gc_files": sum(1 for r in stw_rows if r["missing_gc_log_files"]),

        "total_strict_stw_pause_events_after_warmup": sum(r["stw_pause_count"] for r in stw_rows),
        "total_nonzero_stw_pause_events_after_warmup": sum(r["stw_nonzero_pause_count"] for r in stw_rows),
        "total_zero_duration_stw_pause_events_after_warmup": sum(r["stw_zero_duration_pause_count"] for r in stw_rows),
        "total_stw_ms_after_warmup": sum(r["total_stw_ms"] for r in stw_rows),

        "audit_counters": dict(global_audit_counters),

        "debug_events_written": len(debug_events) if args.debug_stw else 0,

        "errors": errors,
    }

    return throughput_rows, stw_rows, debug_events, validation


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def all_pause_categories(stw_rows: list[dict[str, Any]]) -> list[str]:
    categories = set()

    for row in stw_rows:
        categories.update(row["pause_breakdown"].keys())

    return sorted(categories)


def write_throughput_csv(path: Path, throughput_rows: list[dict[str, Any]]) -> None:
    fields = [
        "benchmark",
        "environment",
        "measurement_count",
        "total_duration_ms",
        "avg_duration_ms",
        "fastest_environment",
        "fastest_avg_duration_ms",
        "is_fastest_environment",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()

        for row in throughput_rows:
            out = {
                "benchmark": row["benchmark"],
                "environment": row["environment"],
                "measurement_count": row["measurement_count"],
                "total_duration_ms": f"{row['total_duration_ms']:.6f}",
                "avg_duration_ms": f"{row['avg_duration_ms']:.6f}",
                "fastest_environment": row["fastest_environment"],
                "fastest_avg_duration_ms": (
                    ""
                    if row["fastest_avg_duration_ms"] == ""
                    else f"{row['fastest_avg_duration_ms']:.6f}"
                ),
                "is_fastest_environment": "TRUE" if row["is_fastest_environment"] else "",
            }

            writer.writerow(out)


def write_stw_csv(path: Path, stw_rows: list[dict[str, Any]]) -> None:
    categories = all_pause_categories(stw_rows)

    category_fields: list[str] = []

    for category in categories:
        slug = category_to_column_slug(category)
        category_fields.extend([
            f"category_{slug}_count",
            f"category_{slug}_ms",
        ])

    fields = [
        "benchmark",
        "environment",
        "processed_gc_log_file_count",

        "stw_pause_count",
        "stw_nonzero_pause_count",
        "stw_zero_duration_pause_count",
        "stw_gc_id_count_with_stw",

        "total_stw_ms",
        "stw_overhead_percent",

        "min_gc_uptime_s_after_warmup",
        "max_gc_uptime_s_after_warmup",

        "audit_accepted_gc_tag",
        "audit_accepted_g1_strict_pause",
        "audit_accepted_zgc_strict_phase",
        "audit_raw_pause_gc_candidates",
        "audit_broad_regex_matched",
        "audit_broad_regex_not_matched",
        "audit_accepted_gc_tag",
        "audit_accepted_zgc_strict_phase",
        "audit_accepted_zgc_relaxed_tag",
        "audit_candidate_zgc_category_but_tag_not_strict",
        "audit_candidate_pause_but_category_or_tag_rejected",
        "audit_rejected_before_warmup",
        "audit_no_duration_token",
        "audit_candidate_zero_duration",
        "audit_candidate_nonzero_duration",
        "audit_duplicate_accepted_event_keys",
    ] + category_fields

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()

        for row in stw_rows:
            audit = row.get("audit_counters", {})

            out: dict[str, Any] = {
                "benchmark": row["benchmark"],
                "environment": row["environment"],
                "processed_gc_log_file_count": row["processed_gc_log_file_count"],

                "stw_pause_count": row["stw_pause_count"],
                "stw_nonzero_pause_count": row["stw_nonzero_pause_count"],
                "stw_zero_duration_pause_count": row["stw_zero_duration_pause_count"],
                "stw_gc_id_count_with_stw": row["stw_gc_id_count_with_stw"],

                "total_stw_ms": f"{row['total_stw_ms']:.6f}",
                "stw_overhead_percent": f"{row['stw_overhead_percent']:.6f}",

                "min_gc_uptime_s_after_warmup": (
                    ""
                    if row["min_gc_uptime_s_after_warmup"] is None
                    else f"{row['min_gc_uptime_s_after_warmup']:.6f}"
                ),
                "max_gc_uptime_s_after_warmup": (
                    ""
                    if row["max_gc_uptime_s_after_warmup"] is None
                    else f"{row['max_gc_uptime_s_after_warmup']:.6f}"
                ),

                "audit_raw_pause_gc_candidates": audit.get("raw_pause_gc_candidates", 0),
                "audit_accepted_gc_tag": audit.get("accepted_gc_tag", 0),
                "audit_accepted_g1_strict_pause": audit.get("accepted_g1_strict_pause", 0),
                "audit_accepted_zgc_strict_phase": audit.get("accepted_zgc_strict_phase", 0),
                "audit_broad_regex_matched": audit.get("broad_regex_matched", 0),
                "audit_broad_regex_not_matched": audit.get("broad_regex_not_matched", 0),
                "audit_accepted_gc_tag": audit.get("accepted_gc_tag", 0),
                "audit_accepted_zgc_strict_phase": audit.get("accepted_zgc_strict_phase", 0),
                "audit_accepted_zgc_relaxed_tag": audit.get("accepted_zgc_relaxed_tag", 0),
                "audit_candidate_zgc_category_but_tag_not_strict": audit.get("candidate_zgc_category_but_tag_not_strict", 0),
                "audit_candidate_pause_but_category_or_tag_rejected": audit.get("candidate_pause_but_category_or_tag_rejected", 0),
                "audit_rejected_before_warmup": audit.get("rejected_before_warmup", 0),
                "audit_no_duration_token": audit.get("no_duration_token", 0),
                "audit_candidate_zero_duration": audit.get("candidate_zero_duration", 0),
                "audit_candidate_nonzero_duration": audit.get("candidate_nonzero_duration", 0),
                "audit_duplicate_accepted_event_keys": audit.get("duplicate_accepted_event_keys", 0),
            }

            for category in categories:
                slug = category_to_column_slug(category)
                bucket = row["pause_breakdown"].get(category, {"count": 0, "total_ms": 0.0})

                out[f"category_{slug}_count"] = bucket["count"]
                out[f"category_{slug}_ms"] = f"{float(bucket['total_ms']):.6f}"

            writer.writerow(out)


def write_stw_debug_csv(path: Path, debug_events: list[dict[str, Any]]) -> None:
    fields = [
        "benchmark",
        "environment",
        "gc_log_file",
        "line_number",
        "status",
        "accepted",
        "uptime_s",
        "gc_id",
        "tags",
        "raw_category",
        "normalized_category",
        "duration_ns",
        "duration_ms",
        "line_hash",
        "duplicate_event_key_count_so_far",
        "raw_line",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()

        for event in debug_events:
            writer.writerow({
                field: event.get(field, "")
                for field in fields
            })


def write_summary_json(
    path: Path,
    throughput_rows: list[dict[str, Any]],
    stw_rows: list[dict[str, Any]],
    warmup_cutoff_ns: int,
    validation: dict[str, Any],
) -> None:
    payload = {
        "warmup_cutoff_ns": warmup_cutoff_ns,
        "throughput_summary": throughput_rows,
        "stw_pause_summary": stw_rows,
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
        description="Aggregate post-warmup Renaissance throughput and auditable JVM STW pause overhead."
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
        help="Benchmark name. Repeat to override the default benchmark list.",
    )

    parser.add_argument(
        "--measurements",
        choices=["auto", "json", "csv"],
        default="auto",
        help="Measurement file type. auto prefers benchmark.json, benchmark.csv, then any JSON/CSV in the benchmark directory.",
    )

    parser.add_argument(
        "--warmup-cutoff-ns",
        type=int,
        default=WARMUP_CUTOFF_NS,
        help="Warmup cutoff in nanoseconds. Default: 5 minutes.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output CSV/JSON files.",
    )

    parser.add_argument(
        "--debug-stw",
        action="store_true",
        help="Write candidate and accepted STW pause events to stw_pause_events_debug.csv.",
    )

    parser.add_argument(
        "--debug-benchmark",
        default=None,
        help="When --debug-stw is enabled, only write debug rows for this benchmark.",
    )

    parser.add_argument(
        "--debug-environment",
        default=None,
        help="When --debug-stw is enabled, only write debug rows for this environment.",
    )

    parser.add_argument(
        "--zgc-relaxed-tags",
        action="store_true",
        help=(
            "Relax ZGC acceptance: accept known ZGC STW categories if 'gc' appears "
            "anywhere in the tag list. Use only after inspecting debug CSV."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    args.env = args.env or ENV_DIRS
    args.benchmark = args.benchmark or BENCHMARKS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    throughput_rows, stw_rows, debug_events, validation = build_summaries(args)

    throughput_csv_path = args.output_dir / "throughput_summary_after_warmup.csv"
    stw_csv_path = args.output_dir / "stw_pause_summary_after_warmup.csv"
    json_path = args.output_dir / "stw_overhead_summary.json"
    debug_csv_path = args.output_dir / "stw_pause_events_debug.csv"

    write_throughput_csv(throughput_csv_path, throughput_rows)
    write_stw_csv(stw_csv_path, stw_rows)
    write_summary_json(
        json_path,
        throughput_rows,
        stw_rows,
        args.warmup_cutoff_ns,
        validation,
    )

    if args.debug_stw:
        write_stw_debug_csv(debug_csv_path, debug_events)

    print("Validation summary")
    print(f"  environments processed: {validation['environment_count']}")
    print(f"  benchmarks processed: {validation['benchmark_count']}")
    print(f"  expected benchmark/environment rows: {validation['expected_benchmark_environment_rows']}")
    print(f"  throughput rows written: {validation['throughput_rows_written']}")
    print(f"  STW rows written: {validation['stw_rows_written']}")
    print(f"  processed GC log files: {validation['processed_gc_log_files']}")
    print(f"  missing GC log groups: {validation['missing_gc_log_groups']}")
    print(f"  rows with missing rotated GC files: {validation['rows_with_missing_rotated_gc_files']}")

    print(f"  total strict STW pause events after warmup: {validation['total_strict_stw_pause_events_after_warmup']}")
    print(f"  total nonzero STW pause events after warmup: {validation['total_nonzero_stw_pause_events_after_warmup']}")
    print(f"  total zero-duration STW pause events after warmup: {validation['total_zero_duration_stw_pause_events_after_warmup']}")
    print(f"  total STW ms after warmup: {validation['total_stw_ms_after_warmup']:.6f}")

    print(f"  throughput CSV path: {throughput_csv_path}")
    print(f"  STW CSV path: {stw_csv_path}")
    print(f"  JSON path: {json_path}")

    if args.debug_stw:
        print(f"  STW debug events written: {validation['debug_events_written']}")
        print(f"  STW debug CSV path: {debug_csv_path}")

    audit = validation.get("audit_counters", {})
    if audit:
        print("\nGlobal parser audit counters")
        for key in sorted(audit):
            print(f"  {key}: {audit[key]}")

    if args.zgc_relaxed_tags:
        print(
            "\nWARNING: --zgc-relaxed-tags was enabled. "
            "ZGC tag filtering was relaxed. Use this only if the debug CSV justifies it.",
            file=sys.stderr,
        )

    if validation["errors"]:
        print("\nErrors / skipped rows:", file=sys.stderr)
        for error in validation["errors"]:
            print(f"  - {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())