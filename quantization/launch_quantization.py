#!/usr/bin/env python3
"""One-shot detached production launch; no automatic restart or container removal.

Run with host Python. The manifest and all run inputs except the source model
must reside under run_root. Export and target cache share one HF-home mount so
hard-link cache materialization does not cross Docker mount boundaries.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import uuid

from write_export import _publish_json


def command(manifest_path, manifest, image, hf_home, attempt, *, resume=False):
    manifest_path, hf_home = Path(manifest_path).resolve(strict=True), Path(hf_home).resolve(strict=True)
    root = Path(manifest["run_root"]).resolve(strict=True)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image) or not re.fullmatch(r"[0-9a-f]{32}", attempt):
        raise ValueError("launch requires immutable image ID and valid attempt")
    if not manifest_path.is_relative_to(root):
        raise ValueError("manifest must reside in the run root")
    if root.is_relative_to(hf_home) or hf_home.is_relative_to(root):
        raise ValueError("run root and HF home must be separate mount trees")
    for name in ("corpus", "input_attestation", "source_attestation", "token_file", "export_state"):
        path = Path(manifest[name])
        if not path.is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError(f"{name} must reside in the run root")
    publication = manifest.get("publication", {})
    if publication.get("repo_id") != "wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1":
        raise ValueError("production launch requires public target publication")
    if Path(publication.get("cache_root", "")).resolve() != hf_home / "hub":
        raise ValueError("publication cache must be the standard HF-home hub")
    output, snapshot = Path(manifest["output"]), Path(manifest["snapshot"])
    if (not output.is_absolute() or output.resolve() == hf_home or not output.resolve().is_relative_to(hf_home)
            or output.resolve().is_relative_to(hf_home / "hub")):
        raise ValueError("export must be staged inside HF home, outside its hub cache")
    if (not snapshot.is_absolute() or not snapshot.is_relative_to(hf_home / "hub")
            or snapshot.parent.name != "snapshots" or not snapshot.is_dir()):
        raise ValueError("source must be an existing standard HF snapshot")
    if [slot["image_digest"] for slot in manifest["coordinator_slots"]] != [image, image]:
        raise ValueError("launch image differs from qualified coordinator slots")
    for path in (manifest_path, root, hf_home, snapshot):
        if "," in str(path):
            raise ValueError("Docker bind paths cannot contain commas")
    token = hf_home / "token"
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("HF token must exist with private permissions")
    label = hashlib.sha256(str(root).encode()).hexdigest()
    args = ["docker", "run", "--detach", "--name", "ds41rt-quant-" + attempt,
            "--label", "ds41rt.quant.run=" + label, "--restart=no", "--gpus", "all",
            "--network=host", "--shm-size=16g", "--memory=170g", "--memory-swap=170g",
            "--log-driver=local", "--log-opt=max-size=20m", "--log-opt=max-file=5",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "HF_HOME=" + str(hf_home),
            "--env", "HF_TOKEN_PATH=" + str(token), "--env", "HF_XET_HIGH_PERFORMANCE=1",
            "--mount", f"type=bind,src={root},dst={root}",
            "--mount", f"type=bind,src={hf_home},dst={hf_home}",
            "--mount", f"type=bind,src={snapshot.parent.parent},dst={snapshot.parent.parent},readonly",
            "--mount", f"type=bind,src={token},dst={token},readonly",
            "--mount", f"type=bind,src={manifest_path},dst={manifest_path},readonly",
            "--mount", "type=volume,src=ds41rt-quant-coordinator-jit,dst=/root/.cache/gptqmodel",
            "--entrypoint", "/usr/bin/tini", image, "--", "/opt/glmrt/quant-venv/bin/python",
            "/opt/ds41rt/quantization/session_runner.py", str(manifest_path), "--attempt", attempt]
    if resume:
        args.append("--resume")
    return args, label


def launch(manifest_path, image, hf_home, *, resume=False, execute=subprocess.check_output):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["run_root"])
    attempt = uuid.uuid4().hex
    args, label = command(manifest_path, manifest, image, hf_home, attempt, resume=resume)
    with (root / "launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actual_image = execute(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
        if actual_image != image:
            raise ValueError("Docker image identity differs")
        ids = execute(["docker", "ps", "--all", "--quiet", "--filter", "label=ds41rt.quant.run=" + label], text=True).split()
        if ids:
            containers = json.loads(execute(["docker", "inspect", *ids], text=True))
            if any(item["State"]["Status"] not in {"exited", "dead"} for item in containers):
                raise ValueError("an existing coordinator is not terminal; inspect it, do not relaunch")
        if (ids or (root / "run.sqlite").exists()) and not resume:
            raise ValueError("existing run requires explicit --resume")
        attempts = root / "attempts"
        attempts.mkdir(exist_ok=True)
        intent = dict(attempt=attempt, image=image, manifest=str(manifest_path), resume=resume,
                      command=args, log=str(attempts / (attempt + ".log")))
        _publish_json(attempts / (attempt + "-launch.json"), intent)
        # On an ambiguous Docker error the durable intent gives the exact name
        # to inspect. Never retry here or remove a prior attempt automatically.
        container = execute(args, text=True).strip()
        result = dict(attempt=attempt, container=container, log=intent["log"], run_root=str(root))
        _publish_json(attempts / (attempt + "-container.json"), result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(launch(args.manifest, args.image, args.hf_home, resume=args.resume), indent=2))
