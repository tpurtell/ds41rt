#!/usr/bin/env python3
"""Bounded authenticated EXL3 worker, adapted from ds4rt's reference server.

The identity report describes the live runtime; it is not numerical qualification.
Deployment must bind the supplied image digest to Docker's inspected image ID.
"""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import threading
import traceback

import torch
import gptqmodel
from gptqmodel.utils.exl3_projection_checkpoint import EXL3ProjectionCheckpointStore, canonical_json_bytes
from gptqmodel.utils.exl3_remote import (
    DEFAULT_MAX_BODY_BYTES, REMOTE_CONTRACT, REMOTE_REQUEST_SCHEMA, REMOTE_RESULT_SCHEMA,
    decode_tensor_envelope, encode_tensor_envelope, execute_remote_projection,
)


def runtime_identity(name, image_digest):
    if not name or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest):
        raise ValueError("stable name and inspected image digest are required")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (12, 1):
        raise ValueError("Spark worker requires exactly one GB10/SM121 device")
    root = Path(gptqmodel.__file__).resolve().parent
    source = hashlib.sha256()
    for path in sorted([*root.rglob("*"), *(root.parent / "gptqmodel_ext").rglob("*")]):
        if path.is_file() and path.suffix in {".py", ".cu", ".cpp", ".h", ".cuh"}:
            source.update(str(path.relative_to(root.parent)).encode() + b"\0")
            source.update(hashlib.sha256(path.read_bytes()).digest())
    props = torch.cuda.get_device_properties(0)
    report = dict(schema="ds41rt-exl3-worker-runtime-v1", name=name, image_digest=image_digest,
                  python=sys.version, gil_enabled=sys._is_gil_enabled(),
                  versions={key: importlib.metadata.version(key) for key in ("torch", "triton", "safetensors")},
                  gptqmodel_source_sha256=source.hexdigest(),
                  worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gpu=dict(name=props.name, uuid=str(props.uuid), memory=props.total_memory),
                  tf32=torch.backends.cuda.matmul.allow_tf32)
    return dict(contract=REMOTE_CONTRACT, name=name, image_digest=image_digest,
                preflight_sha256=hashlib.sha256(canonical_json_bytes(report)).hexdigest(), runtime=report)


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, *, identity, token, checkpoint_root,
                 max_body_bytes=DEFAULT_MAX_BODY_BYTES):
        if not token or max_body_bytes <= 0:
            raise ValueError("worker requires authentication and a positive body limit")
        self.identity, self.token = identity, token
        self.max_body_bytes = max_body_bytes
        self.checkpoint_store = EXL3ProjectionCheckpointStore(checkpoint_root)
        self.staging = threading.BoundedSemaphore(2)
        self.quantize_lock = threading.Lock()
        super().__init__(address, WorkerHandler)


class WorkerHandler(BaseHTTPRequestHandler):
    # Close each connection so rejected unread bodies cannot become requests.
    protocol_version = "HTTP/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(120)

    def signed(self, status, payload, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-DS4RT-Signature", hmac.new(self.server.token, payload, hashlib.sha256).hexdigest())
        self.end_headers()
        self.wfile.write(payload)

    def error(self, status, message):
        self.signed(status, canonical_json_bytes(dict(status="error", message=message)))

    def authenticated(self, payload):
        claimed = self.headers.get("X-DS4RT-Signature", "")
        return hmac.compare_digest(claimed,
            hmac.new(self.server.token, payload, hashlib.sha256).hexdigest())

    def do_GET(self):
        if self.path != "/v1/identity":
            return self.error(404, "not found")
        if not self.authenticated(b"GET /v1/identity"):
            return self.error(401, "authentication failed")
        self.signed(200, canonical_json_bytes(self.server.identity))

    def do_POST(self):
        if self.path != "/v1/exl3/quantize":
            return self.error(404, "not found")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self.error(400, "invalid content length")
        if not 0 < length <= self.server.max_body_bytes or self.headers.get("Transfer-Encoding"):
            return self.error(413, "invalid body size or transfer encoding")
        if not self.server.staging.acquire(blocking=False):
            return self.error(503, "worker staging capacity exhausted")
        try:
            payload = self.rfile.read(length)
            if len(payload) != length:
                return self.error(400, "truncated body")
            if not self.authenticated(payload):
                return self.error(401, "authentication failed")
            manifest, tensors = decode_tensor_envelope(payload, max_body_bytes=self.server.max_body_bytes)
            del payload
            if (manifest.get("schema") != REMOTE_REQUEST_SCHEMA
                    or manifest.get("contract") != REMOTE_CONTRACT
                    or not isinstance(manifest.get("request"), dict)):
                raise ValueError("invalid request manifest")
            with self.server.quantize_lock:
                packed, result, hit = execute_remote_projection(request=manifest["request"], tensors=tensors,
                    device="cuda:0", worker_identity=self.server.identity,
                    checkpoint_store=self.server.checkpoint_store)
            response = encode_tensor_envelope(dict(schema=REMOTE_RESULT_SCHEMA, contract=REMOTE_CONTRACT,
                request_sha256=manifest["request"]["request_sha256"], result=result, checkpoint_hit=hit), packed)
            self.signed(200, response, "application/octet-stream")
        except Exception as error:
            traceback.print_exc()
            self.error(400 if isinstance(error, (ValueError, RuntimeError)) else 500,
                       f"{type(error).__name__}: {str(error)[:2048]}")
        finally:
            self.server.staging.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=17841)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--identity-only", action="store_true")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    identity = runtime_identity(args.name, args.image_digest)
    print(json.dumps(identity, sort_keys=True), flush=True)
    if args.identity_only:
        return
    token = os.environ.get("DS41RT_EXL3_WORKER_TOKEN", "").encode()
    server = WorkerServer((args.host, args.port), identity=identity, token=token,
                          checkpoint_root=args.checkpoint_root)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
