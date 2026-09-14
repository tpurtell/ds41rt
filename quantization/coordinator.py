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
from corpus_inputs import prepare_frontiers
from draft_inputs import prepare_draft_frontiers
from draft_anchors import select_anchors


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


def quantize_namespaces(driver, records, attestation, *, main_adapters, draft_adapters,
                        native_kernels, anchor_count=327680, anchor_seed=20260809):
    """Run both quantized namespaces; factories allocate adapters only if needed.

    Each factory returns two independently owned RTX adapters. This entry point
    owns the run lock. It does not deploy workers or export/upload the result.
    """
    with exclusive_run(driver.journal.root):
        corpus = driver._load("inputs/inventory", "inventory")
        if corpus is not None and (corpus["records"] != records or corpus["attestation"] != attestation):
            raise ValueError("coordinator corpus differs from committed input attestation")
        initial = driver._load("inputs/complete", "inputs")
        if initial is None:
            adapters = main_adapters()
            try:
                initial = prepare_frontiers(driver, records, attestation, adapters)
            finally:
                for index in range(len(adapters)):
                    adapters[index].close()
                del adapters
                gc.collect()
        main = run_namespace(driver, "base", initial, native_kernels=native_kernels)
        draft_inventory = driver._load("draft-inputs/inventory", "inventory")
        selection = select_anchors(records, count=anchor_count, seed=anchor_seed)
        if draft_inventory is not None and draft_inventory != dict(selection=selection, main_frontier=main):
            raise ValueError("coordinator draft selection differs from committed handoff")
        draft_initial = driver._load("draft-inputs/complete", "inputs")
        if draft_initial is None:
            adapters = draft_adapters()
            try:
                draft_initial = prepare_draft_frontiers(driver, main, records, adapters,
                                                        count=anchor_count, seed=anchor_seed)
            finally:
                del adapters
                gc.collect()
        draft = run_namespace(driver, "mtp", draft_initial, native_kernels=native_kernels)
        result = dict(main=main, draft=draft, status="namespaces-quantized-export-pending")
        previous = driver._load("quantization/complete", "quantization")
        if previous is None:
            driver._publish("quantization/complete", "quantization", result,
                            ("namespaces/base/complete", "namespaces/mtp/complete"))
        elif previous != result:
            raise ValueError("quantization completion changed")
        return result
