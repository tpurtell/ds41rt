"""One mixed block's durable input -> routed capture -> propagated output DAG.

No cleanup or automatic recovery is performed. Superseded input storage may only
be retired by a later retention policy after the block-complete record commits.
"""
import torch

from gptqmodel.utils.v41_checkpoint import load_frontier, save_frontier, save_routed_batch
from gptqmodel.utils.v41_routed_batch import V41RoutedBatch
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

    @torch.inference_mode()
    def process(self, block, namespace, layer, input_keys, *, input_provenance):
        if namespace not in NAMESPACES or type(layer) is not int or not 0 <= layer < NAMESPACES[namespace][1]:
            raise ValueError("invalid wavefront namespace/layer")
        if block.layer_idx != layer:
            raise ValueError("wavefront block is at the wrong layer")
        if not input_keys or len(set(input_keys)) != len(input_keys):
            raise ValueError("wavefront requires a nonempty distinct input inventory")
        if set(input_provenance) != set(input_keys):
            raise ValueError("wavefront provenance must cover exactly its inputs")
        root = f"blocks/{namespace}/{layer:03d}"
        inventory_key = root + "/input-inventory"
        inventory = dict(keys=tuple(input_keys), provenance=input_provenance)
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
        routed_keys, routed_provenance = [], {}
        for ordinal, input_key in enumerate(input_keys):
            key = f"{root}/routed/batch-{ordinal:06d}"
            provenance = {**self.driver.identity, "artifact": key}
            if self.journal.get(key, verify=False) is None:
                state = self._input(input_key, input_provenance[input_key], layer)
                routed = V41RoutedBatch.from_replay(block, state, self.driver.device)
                path = self.journal.root / (key + ".safetensors")
                save_routed_batch(routed, path, provenance=provenance)
                self.journal.record_file(key, "routed", path, parents=(input_key, inventory_key))
                del routed, state
                self.driver.progress(dict(event="routed_committed", key=key))
            routed_keys.append(key)
            routed_provenance[key] = provenance
        selected_phase = self.driver.run(block, namespace, layer, routed_keys,
                                          routed_provenance=routed_provenance)
        output_keys, output_provenance = [], {}
        for ordinal, input_key in enumerate(input_keys):
            key = f"{root}/outputs/batch-{ordinal:06d}"
            provenance = {**self.driver.identity, "artifact": key}
            if self.journal.get(key, verify=False) is None:
                state = self._input(input_key, input_provenance[input_key], layer)
                outgoing = state.advance(block, self.driver.device)
                if (outgoing.next_layer != layer + 1 or not torch.isfinite(outgoing.hidden).all()
                        or not torch.isfinite(outgoing.pre_mix).all()):
                    raise ValueError("invalid mixed-block output frontier")
                path = self.journal.root / (key + ".safetensors")
                save_frontier(outgoing, path, provenance=provenance)
                self.journal.record_file(key, "replay", path, parents=(input_key, selected_phase))
                del state, outgoing
                self.driver.progress(dict(event="output_committed", key=key))
            else:
                self._input(key, provenance, layer + 1)
            output_keys.append(key)
            output_provenance[key] = provenance
        complete = dict(output_keys=tuple(output_keys), output_provenance=output_provenance,
                        selected_phase=selected_phase, next_layer=layer + 1)
        self.driver._publish(complete_key, "block", complete,
                              (inventory_key, selected_phase, *output_keys))
        self.driver.progress(dict(event="block_committed", key=complete_key))
        return complete
