#!/usr/bin/env python3
"""Build/start the four bounded Spark workers, preserving persistent state.

This is the worker-deployment component, not the complete quantization launcher.
Existing mismatched/stopped containers require explicit recovery; none are removed.
The generated token must stay private. Returned runtime identities still require
numerical qualification before production dispatch.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import tempfile


BASE = "sha256:a70e6af77cd323ae2fe507fbeb5f6353a6fdce3e9247d2d5b9632bf4c92a55ae"
HOSTS = ("ostrich", "dodo", "emu", "kiwi")


def ssh(host, command, **kwargs):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
                          check=True, **kwargs)


def deploy(host, repo, root, token):
    reports = root / "reports"
    # Bind the known base ID locally. Missing bases fail; never pull a substitute.
    ssh(host, f"docker image inspect {BASE} >/dev/null && docker tag {BASE} ds41rt-quant-base:a70e6af77cd3")
    with (reports / f"{host}-worker-deploy-build.log").open("ab") as log:
        archive = subprocess.Popen(["tar", "-czf", "-", "--exclude=__pycache__", "--exclude=*.pyc",
            "docker/Dockerfile.quant-worker", "quantization/exl3_worker.py",
            "third_party/gptqmodel/gptqmodel", "third_party/gptqmodel/gptqmodel_ext"],
            cwd=repo, stdout=subprocess.PIPE, stderr=log)
        try:
            ssh(host, "docker build -f docker/Dockerfile.quant-worker -t ds41rt-quant-worker:dev -",
                stdin=archive.stdout, stdout=log, stderr=log)
        finally:
            archive.stdout.close()
            code = archive.wait()
        if code:
            raise RuntimeError(f"{host}: build context archive failed")
    image = ssh(host, "docker image inspect ds41rt-quant-worker:dev --format '{{.Id}}'",
                capture_output=True, text=True).stdout.strip()
    existing = ssh(host, "docker ps -a --filter name='^/ds41rt-quant-worker$' --format '{{.ID}}'",
                   capture_output=True, text=True).stdout.strip()
    if existing:
        status = json.loads(ssh(host, "docker inspect ds41rt-quant-worker --format '{{json .}}'",
                                capture_output=True, text=True).stdout)
        # Do not print inspect output: it contains the private token environment.
        if status["Image"] != image or not status["State"]["Running"]:
            raise RuntimeError(f"{host}: existing worker requires explicit recovery")
        if f"DS41RT_EXL3_WORKER_TOKEN={token}" not in status["Config"]["Env"]:
            raise RuntimeError(f"{host}: existing worker uses a different authentication token")
    else:
        command = ("read -r ds41rt_worker_token; docker run -d --name ds41rt-quant-worker --gpus all "
                   "--restart=no -p 17841:17841 -v ds41rt-quant-jit:/root/.cache/gptqmodel "
                   "-v ds41rt-quant-worker-state:/state "
                   '-e DS41RT_EXL3_WORKER_TOKEN="$ds41rt_worker_token" '
                   f"{shlex.quote(image)} --name {host} --image-digest {shlex.quote(image)} "
                   "--checkpoint-root /state/checkpoints")
        ssh(host, command, input=token + "\n", capture_output=True, text=True)
    # The service logs its immutable runtime identity after imports. Docker's
    # running state alone is not readiness; callers must authenticate/qualify it.
    container = ssh(host, "docker inspect ds41rt-quant-worker --format '{{.Id}}'",
                    capture_output=True, text=True).stdout.strip()
    return dict(host=host, image_digest=image, container_id=container, port=17841,
                checkpoint_volume="ds41rt-quant-worker-state", jit_volume="ds41rt-quant-jit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    (root / "reports").mkdir(parents=True, exist_ok=True)
    token_path = root / "worker-token"
    try:
        descriptor = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if token_path.is_symlink() or not token_path.is_file() or token_path.stat().st_mode & 0o077:
            raise ValueError("worker token must be a private regular file")
    else:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(secrets.token_hex(32) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    token = token_path.read_text().strip()
    if len(token) != 64 or any(char not in "0123456789abcdef" for char in token):
        raise ValueError("invalid worker token")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    repo = Path(__file__).resolve().parents[1]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(deploy, host, repo, root, token) for host in HOSTS]
        workers = [future.result() for future in futures]
    manifest = dict(schema="ds41rt-worker-deployment-v1", workers=workers,
                    status="deployed-not-numerically-qualified")
    with tempfile.NamedTemporaryFile(mode="w", dir=root, prefix="workers-", delete=False) as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, root / "workers.json")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
