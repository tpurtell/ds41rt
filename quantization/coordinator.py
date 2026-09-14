"""Exclusive, resumable block-loop orchestration for main and dSpark namespaces.

Input creation, dSpark handoff, deployment and export are separate stages. This
loop never labels the whole model complete merely because one namespace ends.
"""
from contextlib import contextmanager
import fcntl
import gc
import os
import time

import torch

from mixed_recipe import NAMESPACES
from retention import retire_block_temporaries
from wavefront import Wavefront


@contextmanager
def exclusive_run(root):
    """Hold for the whole coordinator lifecycle, including input preparation."""
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / "coordinator.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another coordinator owns this run") from error
        yield
    finally:
        os.close(descriptor)


@torch.inference_mode()
def run_namespace(driver, namespace, initial_frontier, *, native_kernels):
    if namespace not in NAMESPACES or initial_frontier.get("next_layer") != 0:
        raise ValueError("namespace requires its own initial layer-zero frontier")
    if not initial_frontier.get("output_keys"):
        raise ValueError("namespace input inventory is empty")
    initial_key = f"namespaces/{namespace}/inputs"
    previous = driver._load(initial_key, "namespace-inputs")
    if previous is None:
        driver._publish(initial_key, "namespace-inputs", initial_frontier, initial_frontier["output_keys"])
    elif previous != initial_frontier:
        raise ValueError("namespace initial frontier changed during recovery")
    wavefront = Wavefront(driver)
    frontier = wavefront.latest(namespace)
    if frontier is not None:
        # A crash after block commitment but before/during retirement is a normal
        # explicit resume boundary. Its current outputs remain intact.
        retire_block_temporaries(driver, namespace, frontier["next_layer"] - 1)
    else:
        frontier = initial_frontier
    source_namespace, layers, _ = NAMESPACES[namespace]
    for layer in range(frontier["next_layer"], layers):
        started = time.monotonic()
        block = driver.source.load_decoded_block(layer, driver.device,
                    native_kernels=native_kernels, namespace=source_namespace)
        try:
            frontier = wavefront.process(block, namespace, layer, frontier["output_keys"],
                                          input_provenance=frontier["output_provenance"])
        finally:
            del block
            gc.collect()
        removed = retire_block_temporaries(driver, namespace, layer)
        driver.progress(dict(event="block_cycle_complete", namespace=namespace, layer=layer,
                             seconds=time.monotonic() - started, retired_bytes=removed,
                             gpu_allocated_bytes=(torch.cuda.memory_allocated(driver.device)
                                                  if driver.device.type == "cuda" else 0)))
    key = f"namespaces/{namespace}/complete"
    existing = driver._load(key, "namespace")
    if existing is None:
        driver._publish(key, "namespace", frontier,
                        (initial_key, f"blocks/{namespace}/{layers - 1:03d}/complete"))
    elif existing != frontier:
        raise ValueError("namespace completion frontier changed")
    return frontier
