"""V4.1 routed-only K3.25 recipe for the fork's causal mixed-tier selector."""
from pathlib import Path


NAMESPACES = {"base": ("layers", 40, 384), "mtp": ("mtp", 3, 128)}
RATIO = {"w1": 3, "w3": 5, "w2": 8}


def mixed_policy(run_state, namespace):
    _, layers, _ = NAMESPACES[namespace]
    return dict(schema="gptqmodel.exl3-inline-mixed", schema_version=1,
                namespace=namespace, base_bits=3, upgrade_bits=4,
                extra_bits=dict(numerator=1, denominator=4), target_bpw="13/4",
                projection_ratio=dict(RATIO),
                score_kind="base-tier-hessian-weighted-relative-error-times-natural-gate-squared-mass-v1",
                tier_plan_root=str(Path(run_state).resolve() / "inline-mixed-tier-plans"),
                logical_layer_start=0, logical_layer_count=layers)


def layer_quotas(namespace, layer):
    _, layers, experts = NAMESPACES[namespace]
    if type(layer) is not int or not 0 <= layer < layers:
        raise ValueError("layer outside V4.1 recipe")
    # Equal-sized gate/up/down matrices; upgrade exactly one quarter of them.
    return {name: experts * 3 * ratio // 64 for name, ratio in RATIO.items()}


def validate_source_tiers(tiers):
    """Require exactly the source routed matrices and exact per-block quotas.

    Keys are checkpoint-native matrix names without `.weight`. This deliberately
    rejects attention/shared/PLE tensors and an omitted dSpark namespace.
    """
    expected = {f"{prefix}.{layer}.ffn.experts.{expert}.{projection}"
                for prefix, layers, experts in NAMESPACES.values()
                for layer in range(layers) for expert in range(experts) for projection in RATIO}
    if set(tiers) != expected:
        raise ValueError("tier map must cover exactly all 47,232 V4.1 routed projections")
    if any(type(value) is not int or value not in (3, 4) for value in tiers.values()):
        raise ValueError("V4.1 routed tiers must be integer K3 or K4")
    for namespace, (prefix, layers, experts) in NAMESPACES.items():
        for layer in range(layers):
            for projection, quota in layer_quotas(namespace, layer).items():
                count = sum(tiers[f"{prefix}.{layer}.ffn.experts.{expert}.{projection}"] == 4
                            for expert in range(experts))
                if count != quota:
                    raise ValueError(f"{namespace} layer {layer} {projection}: {count} K4 != {quota}")
