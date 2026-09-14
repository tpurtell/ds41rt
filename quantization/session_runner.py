"""Container entry point with persistent attempt stdout/stderr and exit status."""
import argparse
import json
import os
from pathlib import Path
import traceback

from write_export import _publish_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.attempt.isalnum():
        raise ValueError("invalid attempt ID")
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["run_root"]) / "attempts"
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / (args.attempt + ".log"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(descriptor, 1)
    os.dup2(descriptor, 2)
    os.close(descriptor)
    status = dict(attempt=args.attempt, status="failed", exit_code=1)
    try:
        from run_quantization import run
        result = run(manifest, resume=args.resume)
        status = dict(attempt=args.attempt, status="finished", exit_code=0, result=result)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        _publish_json(root / (args.attempt + "-exit.json"), status)


if __name__ == "__main__":
    main()
