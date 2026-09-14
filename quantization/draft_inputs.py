"""Durable two-adapter main-output to joint dSpark input handoff."""
from concurrent.futures import ThreadPoolExecutor

import torch
from gptqmodel.utils.v41_checkpoint import load_frontier, save_frontier

from draft_anchors import select_anchors
from mixed_recipe import NAMESPACES


def prepare_draft_frontiers(driver, main_frontier, records, adapters, *, count=327680, seed=20260809):
    if len(adapters) != 2 or adapters[0] is adapters[1]:
        raise ValueError("draft handoff requires two independently owned RTX adapters")
    corpus = driver._load("inputs/inventory", "inventory")
    main = driver._load("namespaces/base/complete", "namespace")
    if (corpus is None or corpus["records"] != records or main is None or main != main_frontier
            or main["next_layer"] != NAMESPACES["base"][1]
            or len(main["output_keys"]) != len(records)):
        raise ValueError("draft handoff requires the complete mixed main corpus in original order")
    selection = select_anchors(records, count=count, seed=seed)
    inventory_key = "draft-inputs/inventory"
    inventory = dict(selection=selection, main_frontier=main_frontier)
    previous = driver._load(inventory_key, "inventory")
    if previous is None:
        driver._publish(inventory_key, "inventory", inventory, ("inputs/inventory", "namespaces/base/complete"))
    elif previous != inventory:
        raise ValueError("draft anchor selection or main frontier changed during recovery")
    keys, provenance, missing = [], {}, []
    journal = driver.journal
    for ordinal, positions in enumerate(selection["positions"]):
        if not positions:
            continue
        key = f"draft-inputs/frontiers/record-{ordinal:06d}"
        prov = {**driver.identity, "artifact": key}
        record = journal.get(key, verify=False)
        if record is None:
            missing.append((ordinal, key))
        else:
            state = load_frontier(journal.root / record["path"], expected_sha256=record["sha256"], expected_provenance=prov)
            if state.next_layer != 0 or not torch.equal(state.kwargs["anchor_positions"], torch.tensor(positions)):
                raise ValueError("draft frontier differs from frozen anchor selection")
            del state
        keys.append(key)
        provenance[key] = prov

    def prepare(ordinal, key, source_record, adapter):
        with torch.inference_mode():
            source_key = main["output_keys"][ordinal]
            state = load_frontier(journal.root / source_record["path"],
                expected_sha256=source_record["sha256"], expected_provenance=main["output_provenance"][source_key])
            if state.next_layer != NAMESPACES["base"][1] or tuple(state.hidden.shape[:2]) != (1, len(records[ordinal]["input_ids"])):
                raise ValueError("main frontier geometry differs from source record")
            positions = torch.tensor(selection["positions"][ordinal], dtype=torch.long)
            result = adapter.prepare_joint(state.target_features,
                torch.tensor([records[ordinal]["input_ids"]], dtype=torch.long), positions=positions)
            if (result.next_layer != 0 or tuple(result.hidden.shape[:2]) != (len(positions), 5)
                    or not torch.isfinite(result.hidden).all() or not torch.isfinite(result.pre_mix).all()
                    or not torch.isfinite(result.kwargs["main_x"]).all()):
                raise ValueError("invalid joint draft input frontier")
            path = journal.root / (key + ".safetensors")
            save_frontier(result, path, provenance=provenance[key])
            return path

    with ThreadPoolExecutor(max_workers=2) as pool:
        for offset in range(0, len(missing), 2):
            pending = []
            for slot, (ordinal, key) in enumerate(missing[offset:offset + 2]):
                source_key = main["output_keys"][ordinal]
                source_record = journal.get(source_key, verify=False)
                if source_record is None or source_record["kind"] != "replay":
                    raise ValueError("draft handoff source must be a committed main replay")
                pending.append((ordinal, key, source_key,
                    pool.submit(prepare, ordinal, key, source_record, adapters[slot])))
            for ordinal, key, source_key, future in pending:
                path = future.result()
                journal.record_file(key, "replay", path, parents=(inventory_key, source_key))
                driver.progress(dict(event="draft_input_committed", key=key,
                                     anchors=len(selection["positions"][ordinal])))
    result = dict(output_keys=tuple(keys), output_provenance=provenance, next_layer=0)
    previous = driver._load("draft-inputs/complete", "inputs")
    if previous is None:
        driver._publish("draft-inputs/complete", "inputs", result, (inventory_key, *keys))
    elif previous != result:
        raise ValueError("draft input completion changed")
    return result
