"""One mixed block's durable input -> routed capture -> propagated output DAG.

No cleanup or automatic recovery is performed. Superseded input storage may only
be retired by a later retention policy after the block-complete record commits.
"""
import torch
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

from gptqmodel.utils.v41_checkpoint import load_frontier, save_frontier, save_routed_batch
from gptqmodel.utils.v41_routed_batch import V41RoutedBatch
from gptqmodel.utils.v41_mixed_replay import install_projection
from mixed_recipe import NAMESPACES


class Wavefront:
    def __init__(self, driver):
        self.driver = driver
        self.journal = driver.journal

    def _input(self, key, provenance, layer):
        record = self.journal.get(key, verify=False)
        if record is None or record["kind"] != "replay":
            raise ValueError("wavefront input must be a committed replay artifact")
        state = load_frontier(self.journal.root / record["path"], expected_sha256=record["sha256"],
                              expected_provenance=provenance)
        if state.next_layer != layer:
            raise ValueError("wavefront input is at the wrong layer")
        return state

    def latest(self, namespace):
        """Find the newest retained frontier without reading retired predecessors.

        Completed markers must form a contiguous prefix. Missing/corrupt newest
        output is a recovery error, never permission to fall back and recompute
        from an older (possibly retired) boundary.
        """
        if namespace not in NAMESPACES:
            raise ValueError("invalid wavefront namespace")
        latest, gap = None, False
        for layer in range(NAMESPACES[namespace][1]):
            complete = self.driver._load(f"blocks/{namespace}/{layer:03d}/complete", "block")
            if complete is None:
                gap = True
                continue
            if gap or complete.get("next_layer") != layer + 1 or not complete.get("output_keys"):
                raise ValueError("completed wavefront markers are not a valid contiguous prefix")
            latest = complete
        if latest is not None:
            for key in latest["output_keys"]:
                self._input(key, latest["output_provenance"][key], latest["next_layer"])
        return latest

    def _activate(self, blocks, root, layer, input_keys, input_provenance, *, routed, parent):
        keys = [f"{root}/{'routed' if routed else 'outputs'}/batch-{ordinal:06d}"
                for ordinal in range(len(input_keys))]
        provenance = {key: {**self.driver.identity, "artifact": key} for key in keys}

        def execute(ordinal, record, block, device):
            scope = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
            with torch.inference_mode(), scope:
                state = load_frontier(self.journal.root / record["path"],
                    expected_sha256=record["sha256"], expected_provenance=input_provenance[input_keys[ordinal]])
                if state.next_layer != layer:
                    raise ValueError("activation input is at the wrong layer")
                key = keys[ordinal]
                path = self.journal.root / (key + ".safetensors")
                if routed:
                    result = V41RoutedBatch.from_replay(block, state, device)
                    save_routed_batch(result, path, provenance=provenance[key])
                else:
                    result = state.advance(block, device)
                    if (result.next_layer != layer + 1 or not torch.isfinite(result.hidden).all()
                            or not torch.isfinite(result.pre_mix).all()):
                        raise ValueError("invalid mixed-block output frontier")
                    save_frontier(result, path, provenance=provenance[key])
                return path

        # Windows use original ordinals, not the remaining-job list: a resumed
        # batch stays on its original device and each block has one active call.
        with ThreadPoolExecutor(max_workers=len(blocks)) as pool:
            for offset in range(0, len(input_keys), len(blocks)):
                pending = []
                for ordinal in range(offset, min(offset + len(blocks), len(input_keys))):
                    key = keys[ordinal]
                    existing = self.journal.get(key, verify=False)
                    if existing is not None:
                        if existing["kind"] != ("routed" if routed else "replay"):
                            raise ValueError("activation artifact kind changed")
                        if not routed:
                            self._input(key, provenance[key], layer + 1)
                        continue
                    record = self.journal.get(input_keys[ordinal], verify=False)
                    if record is None or record["kind"] != "replay":
                        raise ValueError("activation input must be a committed replay")
                    block, device = blocks[ordinal % len(blocks)]
                    pending.append((ordinal, pool.submit(execute, ordinal, record, block, device)))
                for ordinal, future in pending:
                    path = future.result()
                    self.journal.record_file(keys[ordinal], "routed" if routed else "replay", path,
                                             parents=(input_keys[ordinal], parent))
                    self.driver.progress(dict(event="routed_committed" if routed else "output_committed",
                                               key=keys[ordinal]))
        return keys, provenance

    @torch.inference_mode()
    def process(self, block, namespace, layer, input_keys, *, input_provenance, replica=None):
        if namespace not in NAMESPACES or type(layer) is not int or not 0 <= layer < NAMESPACES[namespace][1]:
            raise ValueError("invalid wavefront namespace/layer")
        if block.layer_idx != layer:
            raise ValueError("wavefront block is at the wrong layer")
        if not input_keys or len(set(input_keys)) != len(input_keys):
            raise ValueError("wavefront requires a nonempty distinct input inventory")
        if set(input_provenance) != set(input_keys):
            raise ValueError("wavefront provenance must cover exactly its inputs")
        blocks = [(block, self.driver.device)]
        if replica is not None:
            other, device = replica
            device = torch.device(device)
            if other is block or other.layer_idx != layer or device == self.driver.device:
                raise ValueError("activation replica must own a distinct block and device")
            if {device, self.driver.device} != {torch.device("cuda:0"), torch.device("cuda:1")}:
                raise ValueError("parallel activations require exactly the two RTX devices")
            blocks.append((other, device))
        root = f"blocks/{namespace}/{layer:03d}"
        inventory_key = root + "/input-inventory"
        inventory = dict(keys=tuple(input_keys), provenance=input_provenance,
                         activation_devices=tuple(str(device) for _, device in blocks))
        previous = self.driver._load(inventory_key, "inventory")
        if previous is None:
            self.driver._publish(inventory_key, "inventory", inventory, tuple(input_keys))
        elif previous != inventory:
            raise ValueError("wavefront input inventory changed during recovery")
        complete_key = root + "/complete"
        complete = self.driver._load(complete_key, "block")
        if complete is not None:
            for key in complete["output_keys"]:
                self._input(key, complete["output_provenance"][key], layer + 1)
            return complete
        routed_keys, routed_provenance = self._activate(blocks, root, layer, input_keys, input_provenance,
                                                        routed=True, parent=inventory_key)
        selected_phase = self.driver.run(block, namespace, layer, routed_keys,
                                          routed_provenance=routed_provenance)
        if replica is not None:
            for phase in ("gate_up", "down"):
                marker = self.driver._load(root + f"/{phase}/complete", "phase")
                for expert, projection, key in marker["selected"]:
                    candidate = self.driver._load(key, "projection")
                    install_projection(other, expert, projection, candidate["packed"], device=device)
        output_keys, output_provenance = self._activate(blocks, root, layer, input_keys, input_provenance,
                                                        routed=False, parent=selected_phase)
        complete = dict(output_keys=tuple(output_keys), output_provenance=output_provenance,
                        selected_phase=selected_phase, next_layer=layer + 1)
        self.driver._publish(complete_key, "block", complete,
                              (inventory_key, selected_phase, *output_keys))
        self.driver.progress(dict(event="block_committed", key=complete_key))
        return complete
