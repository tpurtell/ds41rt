#!/usr/bin/env python3
"""Install one measured v6 table block in README and the performance report."""

import argparse
import json
from pathlib import Path


START = "## Performance\n"
END = "## Getting started\n"


def read(path):
    return path.read_text()


def update_readme(current, tables):
    assert current.count(START) == current.count(END) == 1
    before, remainder = current.split(START, 1)
    _, after = remainder.split(END, 1)
    intro = (
        "The official full checkpoint remains the default. Every performance cell contains "
        "three samples. Counting is a low-entropy reference; weighted content, reasoning code, "
        "topic, mixed traffic, and retained-context decode are the primary serving measurements. "
        "See the [v6 performance report](docs/release-v6-performance.md) for the protocol and "
        "provenance.\n\n"
    )
    return before + START + "\n" + intro + tables.rstrip() + "\n\n" + END + after


def performance_report(summary, tables):
    assert summary["release"] == "v6" and summary["performance_matrix_passed"]
    provenance = (
        f"Engine `{summary['engine_commit']}`; SparkInfer `{summary['sparkinfer_commit']}`; "
        f"official checkpoint `{summary['model_id']}@{summary['model_revision']}`. "
        "The [structured summary](release-v6-performance.json) records commands, controls, "
        "artifact hashes, deployment capacity, and telemetry.\n\n"
    )
    return (
        "# DS41RT v6 performance\n\n"
        "These are the native full-checkpoint measurements used to qualify v6. "
        "Historical EXL3 results are linked below and are not v6 measurements.\n\n"
        + provenance + tables.rstrip() + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report already exists; preserve published evidence")
    summary = json.loads(read(args.summary))
    tables = read(args.tables)
    assert tables.startswith("RTX measurements use **400 W per card")
    args.readme.write_text(update_readme(read(args.readme), tables))
    args.report.write_text(performance_report(summary, tables))


if __name__ == "__main__":
    main()
