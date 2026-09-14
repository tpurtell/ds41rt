"""V4.1 projection adapter for the fork's durable authenticated EXL3 scheduler.

Only weights and raw recovered Hessians leave the RTX coordinator. No remote
activation generation or V4.1 model loading is involved.
"""
import hashlib

import torch

from gptqmodel.utils.exl3_projection_checkpoint import build_projection_request, canonical_json_bytes
from gptqmodel.utils.exl3_remote import (
    CoordinatorSlot, EXL3_HESSIAN_CAPTURE_CONTRACT, EXL3_HESSIAN_NUMERICAL_CONTRACT,
    EXL3_HESSIAN_SYMMETRY_CONTRACT, validate_exl3_hessian_metrics,
)
from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3


class DistributedSearch:
    def __init__(self, source, client, identity):
        if len(client.coordinator_slots) != 2 or len(client.endpoints) != 4:
            raise ValueError("V4.1 search requires the two RTX slots and four Spark endpoints")
        if client.assignment_store_path is None:
            raise ValueError("distributed search requires durable slot assignments")
        self.source, self.client, self.identity = source, client, identity
        self.max_workers = len(client.coordinator_slots) + 2 * len(client.endpoints)

    @torch.inference_mode()
    def __call__(self, name, hessian, bits):
        if bits not in (3, 4) or hessian.get("finalized") is not False:
            raise ValueError("V4.1 distributed search requires K3/K4 and raw Hessians")
        assignment = hashlib.sha256(canonical_json_bytes(dict(
            run=self.identity, module=name, bits=bits))).hexdigest()
        lease = self.client.acquire_slot(assignment)
        try:
            slot = lease.slot
            device = slot.device if isinstance(slot, CoordinatorSlot) else "cpu"
            # Match the local driver's BF16 source decode followed by FP32 search.
            weight = self.source.decoded(name + ".weight", device).T.contiguous().float()
            contract = dict(bits=bits, codebook="mcg", apply_out_scales=None, sigma_reg=.025,
                            seed=787, hessian_capture=EXL3_HESSIAN_CAPTURE_CONTRACT,
                            hessian_numerical=EXL3_HESSIAN_NUMERICAL_CONTRACT,
                            hessian_symmetry=EXL3_HESSIAN_SYMMETRY_CONTRACT,
                            execution=self.client.execution_contract(slot))
            if isinstance(slot, CoordinatorSlot):
                args = dict(K=bits, devices=[torch.device(device)], apply_out_scales=None,
                            sigma_reg=.025, seed=787, mcg=True)
                _, _, packed = quantize_exl3(weight,
                    {**hessian, "H": hessian["H"].to(device, copy=True)}, args, return_weight_q=False)
                if args.get("q_fallback") is not False:
                    raise RuntimeError("coordinator search returned a fallback")
                metrics = args["error_metrics"]
            else:
                request = build_projection_request(module_full_name=name,
                    layer_index=int(name.split(".")[1]), input_weight=weight,
                    hessian=hessian["H"], sample_count=hessian["count"],
                    quantizer_contract=contract, family_join={"run": self.identity}, route_evidence=None)
                packed, result, transport = self.client.quantize(endpoint=slot,
                    request_manifest=request, input_weight=weight, hessian=hessian["H"])
                metrics = result["quantizer_metrics"]
                metrics = {**metrics, "remote_transport": transport}
            validate_exl3_hessian_metrics(metrics, sample_count=hessian["count"], sigma_reg=.025)
            return packed, {**metrics, "execution": contract["execution"], "assignment": assignment}
        finally:
            lease.release()
