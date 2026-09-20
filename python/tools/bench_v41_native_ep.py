#!/usr/bin/env python3
"""Tracked entry point for the native replicated-group M1 timing crosscheck.

The measurement logic lives in ``qualify_v41_replicated_native.py`` (imported
here); this file exists so the timing harness has a stable, named, reviewable
source and a single reproducible command. It never edits a kernel.

Example (DODO GB10, frozen lib):

    PYTHONPATH=python/tools:third_party/sparkinfer:python \\
    CUDA_VISIBLE_DEVICES=<uuid> python3 python/tools/bench_v41_native_ep.py \\
      --native-lib <libds41rt_native.so> --spark-tp 2 --layer 20 --tp-rank 0 \\
      --expert-ids 0,1,2,3,4,5 --capacities 1,80 --active-counts 6,3,2 \\
      --no-repack-compare

Use ``--spark-tp 3`` for TP3 or ``--tp4-legacy`` for the role-1 TP4 baseline.
"""

from __future__ import annotations

import sys

from qualify_v41_replicated_native import main


if __name__ == "__main__":
    if "--timing" not in sys.argv and "--checkpoint" not in sys.argv:
        sys.argv.append("--timing")
    raise SystemExit(main())
