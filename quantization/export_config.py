"""Portable EXL3 discovery/storage metadata with explicit native-weight scope."""
import copy

from gptqmodel.quantization.config import EXL3Config
from mixed_recipe import validate_source_tiers


def model_metadata(source_config, inventory, *, provenance):
    if source_config.get("model_type") != "deepseek_v41" or not provenance:
        raise ValueError("V4.1 source config and export provenance are required")
    tiers, tensors = inventory["tiers"], inventory["tensors"]
    validate_source_tiers(tiers)
    storage = {}
    dtypes = {"I16": "int16", "I32": "int32", "F16": "float16"}
    for name, bits in sorted(tiers.items()):
        stored = {}
        inputs, outputs = (2304, 5120) if name.endswith(".w2") else (5120, 2304)
        expected = {"trellis": ("I16", [inputs // 16, outputs // 16, bits * 16]),
                    "suh": ("F16", [inputs]), "svh": ("F16", [outputs]), "mcg": ("I32", [])}
        if name + ".weight" in tensors or name + ".scale" in tensors:
            raise ValueError("native source projection remains in EXL3 inventory")
        for suffix, (dtype, shape) in expected.items():
            key = name + "." + suffix
            item = tensors[key]
            if (item["dtype"] != dtype or (item["shape"] != shape and not (suffix == "mcg" and item["shape"] == [1]))):
                raise ValueError("EXL3 storage descriptor differs from selected tier")
            stored[key] = dict(shape=item["shape"], torch_dtype=dtypes[dtype])
        storage[name] = dict(quant_format="exl3", bits_per_weight=bits, stored_tensors=stored)
    external = EXL3Config(bits=3, codebook="mcg", out_scales="never", tensor_storage=storage, offload_to_disk=False,
                          module_include=[r"^(?:layers|mtp)\.\d+\.ffn\.experts\.\d+\.w[123]$"]).to_dict()
    external.setdefault("meta", {})["ds41rt"] = dict(
        schema="ds41rt.v41-routed-exl3.v1", tensor_naming="checkpoint-native",
        routed_average_bpw="13/4", projection_ratio=dict(w1=3, w3=5, w2=8),
        native_quantization_config=copy.deepcopy(source_config.get("quantization_config")),
        provenance=copy.deepcopy(provenance),
        loader_validation="deferred-to-inference-engine-integration")
    config = copy.deepcopy(source_config)
    config["quantization_config"] = {key: external[key] for key in
                                    ("quant_method", "format", "checkpoint_format", "bits")}
    return {"config.json": config, "quantize_config.json": external}
