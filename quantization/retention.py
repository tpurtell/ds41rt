"""Retire superseded wavefront payloads only behind verified block completion."""
import torch
from gptqmodel.utils.v41_checkpoint import load_frontier

from mixed_recipe import NAMESPACES


def retire_block_temporaries(driver, namespace, layer):
    if namespace not in NAMESPACES or type(layer) is not int or not 0 <= layer < NAMESPACES[namespace][1]:
        raise ValueError("invalid retention block")
    root = f"blocks/{namespace}/{layer:03d}"
    barrier = root + "/complete"
    complete = driver._load(barrier, "block")
    if complete is None:
        raise ValueError("cannot retire an incomplete block")
    journal = driver.journal
    # Verify every replacement frontier, one batch at a time, before authorizing
    # any deletion. This intentionally performs disk reads at the commit barrier.
    for key in complete["output_keys"]:
        record = journal.get(key)
        if record["kind"] != "replay":
            raise ValueError("retention barrier has an invalid output kind")
        state = load_frontier(journal.root / record["path"], expected_sha256=record["sha256"],
                              expected_provenance=complete["output_provenance"][key])
        if state.next_layer != layer + 1 or not torch.isfinite(state.hidden).all() or not torch.isfinite(state.pre_mix).all():
            raise ValueError("retention barrier has an invalid output frontier")
        del state
    selected = set()
    for phase in ("gate_up", "down"):
        marker = driver._load(root + f"/{phase}/complete", "phase")
        if marker is None:
            raise ValueError("retention barrier lacks a selected phase")
        for _, _, key in marker["selected"]:
            if driver._load(key, "projection") is None:
                raise ValueError("retention barrier lacks a selected projection")
            selected.add(key)
    inputs = driver._load(root + "/input-inventory", "inventory")
    if inputs is None or not complete["output_keys"] or set(inputs["keys"]) & set(complete["output_keys"]):
        raise ValueError("retention requires distinct complete input/output inventories")
    keys = list(inputs["keys"])
    # SQL prefix matching is escaped structurally by the fixed namespace/index.
    for key, in journal.db.execute("SELECT key FROM artifacts WHERE key LIKE ? ORDER BY key", (root + "/%",)):
        record = journal.get(key, verify=False)
        if record["kind"] in {"routed", "hessian"} or (record["kind"] == "projection" and key not in selected):
            keys.append(key)
    if selected.intersection(keys) or set(complete["output_keys"]).intersection(keys):
        raise ValueError("retention would remove a selected weight or current frontier")
    removed = journal.retire_files(keys, barrier=barrier)
    driver.progress(dict(event="temporaries_retired", block=barrier, payloads=len(keys), removed_bytes=removed))
    return removed
