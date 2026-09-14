"""Journaled causal K3/K4 block phases over immutable routed batches.

This local search implementation is also the reference for distributed dispatch.
It stops on failure; calling it again is an explicit, identity-checked recovery.
It does not create corpus frontiers, propagate outputs, export, or launch workers.
"""
import gc

import torch

from gptqmodel.utils.v41_capture import V41Capture
from gptqmodel.utils.v41_recovery import V41Recovery
from gptqmodel.utils.v41_checkpoint import _save_state, _load_state, load_routed_batch
from gptqmodel.utils.v41_mixed_replay import install_projection
from gptqmodel.utils.exl3_inline_mixed import projection_score
from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

from mixed_recipe import NAMESPACES, layer_quotas


class BlockDriver:
    def __init__(self, source, journal, identity, *, device, subset_size=8, search=None, progress=None):
        if type(subset_size) is not int or subset_size < 1:
            raise ValueError("expert subset size must be positive")
        self.source, self.journal, self.identity = source, journal, identity
        from run_store import canonical
        if journal.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0] != canonical(identity):
            raise ValueError("block driver identity differs from journal")
        self.device = torch.device(device)
        self.subset_size = subset_size
        self.search = search or self._search
        self.progress = progress or (lambda event: None)

    def _load(self, key, kind):
        record = self.journal.get(key, verify=False)
        if record is None:
            return None
        if record["kind"] != kind:
            raise ValueError("journal artifact kind differs from requested phase")
        return _load_state(self.journal.root / record["path"], expected_sha256=record["sha256"],
                           expected_provenance={**self.identity, "artifact": key}, kind=kind)

    def _publish(self, key, kind, state, parents):
        path = self.journal.root / (key + ".safetensors")
        _save_state(state, path, provenance={**self.identity, "artifact": key}, kind=kind)
        self.journal.record_file(key, kind, path, parents=parents)

    def _search(self, source_name, hessian, bits):
        weight = self.source.decoded(source_name + ".weight", self.device).T.contiguous().float()
        hessian = {**hessian, "H": hessian["H"].to(self.device, copy=True)}
        args = dict(K=bits, devices=[self.device], apply_out_scales=None, sigma_reg=.025, seed=787, mcg=True)
        _, _, packed = quantize_exl3(weight, hessian, args, return_weight_q=False)
        return packed, args["error_metrics"]

    def _candidate(self, prefix, expert, projection, bits, hessian_key, source_prefix):
        key = f"{prefix}/expert-{expert:03d}/{projection}-k{bits}"
        result = self._load(key, "projection")
        if result is None:
            captured = self._load(hessian_key, "hessian")
            packed, metrics = self.search(f"{source_prefix}.ffn.experts.{expert}.{projection}",
                                           captured["hessian"], bits)
            result = dict(packed=packed, quantizer_metrics=metrics, route_evidence=captured["evidence"])
            # Fail before commitment if a scored candidate lacks valid evidence.
            projection_score(result)
            self._publish(key, "projection", result, (hessian_key,))
            result = self._load(key, "projection")
            self.progress(dict(event="candidate_committed", key=key, bits=bits))
        return key, result

    def run(self, block, namespace, layer, routed_keys, *, routed_provenance):
        source_namespace, layers, experts = NAMESPACES[namespace]
        if (block.layer_idx != layer or not 0 <= layer < layers
                or len(block.mlp.experts) != experts or not routed_keys
                or len(set(routed_keys)) != len(routed_keys)):
            raise ValueError("block or routed batch inventory differs from recipe")
        if self.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("block capture requires TF32 disabled at process startup")
        root = f"blocks/{namespace}/{layer:03d}"
        source_prefix = f"{source_namespace}.{layer}"
        # Bind the ordered batch inventory before deriving any candidates.
        inventory_key = root + "/routed-inventory"
        inventory = dict(keys=tuple(routed_keys), provenance=routed_provenance)
        existing = self._load(inventory_key, "inventory")
        if existing is None:
            self._publish(inventory_key, "inventory", inventory, tuple(routed_keys))
        elif existing != inventory:
            raise ValueError("routed corpus inventory changed during recovery")
        parent = inventory_key
        for phase, projections in (("gate_up", ("w1", "w3")), ("down", ("w2",))):
            prefix = f"{root}/{phase}"
            complete_key = prefix + "/complete"
            complete = self._load(complete_key, "phase")
            if complete is not None:
                for expert, projection, key in complete["selected"]:
                    candidate = self._load(key, "projection")
                    install_projection(block, expert, projection, candidate["packed"], device=self.device)
                parent = complete_key
                continue
            for start in range(0, experts, self.subset_size):
                subset = list(range(start, min(experts, start + self.subset_size)))
                missing = [index for index in subset
                           if self.journal.get(f"{prefix}/expert-{index:03d}/hessian", verify=False) is None]
                if missing:
                    capture = V41Capture(block, missing, device=self.device, phase=phase)
                    recovery = V41Recovery(block, missing)
                    for batch_key in routed_keys:
                        record = self.journal.get(batch_key, verify=False)
                        batch = load_routed_batch(self.journal.root / record["path"],
                                                  expected_sha256=record["sha256"],
                                                  expected_provenance=routed_provenance[batch_key])
                        capture.capture_routed(batch)
                        recovery.observe_routed(batch)
                    for expert in missing:
                        hessian, evidence = recovery.projection(capture, expert, projections[0])
                        key = f"{prefix}/expert-{expert:03d}/hessian"
                        self._publish(key, "hessian", dict(hessian=hessian, evidence=evidence), (parent,))
                    del capture, recovery, batch, hessian
                    gc.collect()
                    self.progress(dict(event="capture_committed", phase=phase, experts=missing))
                for expert in subset:
                    hessian_key = f"{prefix}/expert-{expert:03d}/hessian"
                    for projection in projections:
                        self._candidate(prefix, expert, projection, 3, hessian_key, source_prefix)
            # Choose K4 by K3 risk within each projection quota. Tie-break by
            # numeric expert index, frozen as part of this driver's contract.
            tier_key = prefix + "/tiers"
            tier = self._load(tier_key, "tiers")
            if tier is None:
                upgrades, candidate_keys = {}, []
                for projection in projections:
                    scores = []
                    for expert in range(experts):
                        key = f"{prefix}/expert-{expert:03d}/{projection}-k3"
                        scores.append((-projection_score(self._load(key, "projection")), expert))
                        candidate_keys.append(key)
                    scores.sort()
                    upgrades[projection] = tuple(expert for _, expert in scores[:layer_quotas(namespace, layer)[projection]])
                tier = dict(upgrades=upgrades)
                self._publish(tier_key, "tiers", tier, tuple(candidate_keys))
            selected = []
            for expert in range(experts):
                for projection in projections:
                    bits = 4 if expert in tier["upgrades"][projection] else 3
                    key, candidate = self._candidate(prefix, expert, projection, bits,
                                                     f"{prefix}/expert-{expert:03d}/hessian", source_prefix)
                    install_projection(block, expert, projection, candidate["packed"], device=self.device)
                    selected.append((expert, projection, key))
            self._publish(complete_key, "phase", dict(selected=tuple(selected)),
                          (tier_key, *(key for _, _, key in selected)))
            parent = complete_key
            self.progress(dict(event="phase_committed", key=complete_key))
        return parent
