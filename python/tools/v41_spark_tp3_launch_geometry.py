#!/usr/bin/env python3
"""Launch-geometry identity for the native Spark TP3 slice expert kernel.

The native ``spark_tp3`` FP8-activation artifact is exported by
``python/tools/export_b12x_v41_slices_aot.py`` from the pinned
``b12x.moe._shared.kernels.w4a8_v41_slice.V41FusedSliceKernel`` plus
``b12x.moe._shared.kernels.v41_route_plan.V41RoutePlan``. This module is the one
place that turns the artifact's planned slice width into the geometry identity a
qualification record must carry.

What the geometry actually is (and is not):

* **fc1** fuses ``w1`` and ``w3`` into one ``w13`` bank. One CTA covers ``width``
  intermediate columns for *each* projection, so the logical per-projection N is
  ``width`` while the fused ``w13`` extent is ``2 * width``. The M extent is the
  fixed 16-row route group; K is the 5120-wide hidden axis staged as
  ``hidden / 128 == 40`` stages of 128 elements on the ``m16n8k32`` MMA atom.
* **fc2** is token-major: N is the 5120-wide hidden axis in 40 stages of 128, K
  is this slice's intermediate contribution (``width``) in ``width / 32`` atoms.
* The route metadata is ``[expert, row_count, route_base, input_row_ids[16]]``,
  i.e. extent ``3 + M``.
* ``output_kind`` is the manifest's compiled output (``fp32_tokens`` for the
  atomic ABI-3 variants, ``fp32_routes`` for ABI 2).

M (16) and the K32 atom / K128 stage are pinned kernel constants. There is no M
or K plan control in the current artifact, and this module does **not** invent
one: the identity records ``m_control``/``k_control`` as
``pinned_kernel_constant`` and ``n_control`` as ``artifact_manifest_width``.
When an upstream pin exposes typed M/K plan inputs, that config is threaded in
as an explicit field here rather than guessed.

Only static model geometry and planned capacity enter the identity. No live row,
token, group, or expert count is ever part of it, so it can key an export or a
cache without leaking request quantities.
"""

from __future__ import annotations

from typing import Any, Mapping

SPARK_TP3_ROLE = "spark_tp3"
SPARK_TP3_TP_DEGREE = 3
SUPPORTED_SLICE_WIDTHS = (64, 128, 192)
FC1_MMA_ATOM_MNK = (16, 8, 32)
ROUTE_GROUP_ROWS = 16
K_ATOM_ELEMENTS = 32
K_STAGE_ELEMENTS = 128
HIDDEN_ELEMENTS = 5120
BLOCK_THREADS = 128
METADATA_PREFIX_COLUMNS = 3
OUTPUT_KINDS = ("fp32_tokens", "fp32_routes")
#: Bump when the serialized identity schema or its semantics change.
GEOMETRY_SCHEMA_VERSION = 2

GEOMETRY_SOURCE = (
    "pinned sparkinfer w4a8_v41_slice.py::V41FusedSliceKernel + "
    "v41_route_plan.py::V41RoutePlan"
)


class LaunchGeometryError(ValueError):
    """Malformed, unsupported or inconsistent launch geometry."""


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LaunchGeometryError(f"{name} must be a positive int, got {value!r}")
    return value


def launch_geometry(
    capacity: int,
    width: int,
    *,
    intermediate: int,
    kernel_intermediate: int,
    output_kind: str,
    hidden: int = HIDDEN_ELEMENTS,
    revision: str | None = None,
) -> dict[str, Any]:
    """Resolve the executed geometry identity for one compiled variant.

    Keyed on planned capacity and static model geometry only. Fails closed for
    any width/slice/padding combination the pinned kernel does not accept.
    """
    _positive_int("capacity", capacity)
    _positive_int("intermediate", intermediate)
    _positive_int("kernel_intermediate", kernel_intermediate)
    _positive_int("hidden", hidden)
    if width not in SUPPORTED_SLICE_WIDTHS:
        raise LaunchGeometryError(
            f"width {width!r} is not one of {SUPPORTED_SLICE_WIDTHS}; the pinned "
            "V41FusedSliceKernel asserts exactly these"
        )
    if output_kind not in OUTPUT_KINDS:
        raise LaunchGeometryError(
            f"output_kind {output_kind!r} is not one of {OUTPUT_KINDS}"
        )
    # The role geometry is pinned: a hidden axis other than 5120, or a storage
    # extent that is not exactly the 128-element roundup of the logical
    # intermediate, is a different kernel contract and must fail closed.
    if hidden != HIDDEN_ELEMENTS:
        raise LaunchGeometryError(
            f"hidden {hidden} != pinned hidden {HIDDEN_ELEMENTS}; the Spark TP3 "
            "slice kernel has no other hidden extent"
        )
    expected_kernel_intermediate = (
        intermediate + K_STAGE_ELEMENTS - 1) // K_STAGE_ELEMENTS * K_STAGE_ELEMENTS
    if kernel_intermediate != expected_kernel_intermediate:
        raise LaunchGeometryError(
            f"kernel_intermediate {kernel_intermediate} != 128-roundup "
            f"{expected_kernel_intermediate} of intermediate {intermediate}"
        )
    slices = (intermediate + width - 1) // width
    if slices * width > kernel_intermediate:
        raise LaunchGeometryError(
            f"slice tiling {slices} x {width} exceeds kernel_intermediate "
            f"{kernel_intermediate}"
        )
    k_stages = hidden // K_STAGE_ELEMENTS
    source = GEOMETRY_SOURCE + (f" @{revision}" if revision else "")
    return {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "role": SPARK_TP3_ROLE,
        "capacity": capacity,
        "slice_width": width,
        "intermediate": intermediate,
        "kernel_intermediate": kernel_intermediate,
        "slices": slices,
        "fc1": {
            "projections": ["w1", "w3"],
            "logical_n_per_projection": width,
            "fused_w13_n": 2 * width,
            "m_rows": ROUTE_GROUP_ROWS,
            "k_elements": hidden,
            "k_mma_atom": K_ATOM_ELEMENTS,
            "k_stage_elements": K_STAGE_ELEMENTS,
            "k_stages": k_stages,
            "mma_atom_mnk": list(FC1_MMA_ATOM_MNK),
        },
        "fc2": {
            "m_rows": ROUTE_GROUP_ROWS,
            "n_elements": hidden,
            "n_stage_elements": K_STAGE_ELEMENTS,
            "n_stages": k_stages,
            "k_elements": width,
            "k_mma_atoms": width // K_ATOM_ELEMENTS,
        },
        "metadata": {
            "columns": METADATA_PREFIX_COLUMNS + ROUTE_GROUP_ROWS,
            "layout": ["expert", "row_count", "route_base", "input_row_ids[16]"],
        },
        "output_kind": output_kind,
        "grid": {
            "x_slices": slices,
            "y": "runtime_route_group_count",
            "block_threads": BLOCK_THREADS,
            "warps": BLOCK_THREADS // 32,
        },
        # Per-stage control authority. M and K are pinned kernel constants at
        # every stage; the only planned axis is the slice width, which sets fc1
        # N (per projection) and fc2 K (this slice's reduction contribution).
        "controls": {
            "fc1": {"m": "pinned_kernel_constant",
                    "n": "artifact_manifest_slice_width",
                    "k": "pinned_kernel_constant"},
            "fc2": {"m": "pinned_kernel_constant",
                    "n": "pinned_kernel_constant",
                    "k": "artifact_manifest_slice_width"},
            "grid": {"x_slices": "derived_from_manifest_slice_width",
                     "y": "runtime_route_group_count",
                     "block": "pinned_kernel_constant"},
        },
        "source": source,
    }


def manifest_geometry(
    manifest: Mapping[str, Any],
    *,
    degree: int,
    intermediate: int,
    revision: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Resolve every variant's geometry and verify any recorded identity.

    Returns ``{capacity: geometry}``. A manifest that records a ``launch_geometry``
    block which disagrees with the resolved identity fails closed, so a stale or
    hand-edited artifact cannot be qualified under a geometry it never compiled.
    """
    geometry = manifest.get("geometry") or {}
    if manifest.get("spark_tp_degree") != degree:
        raise LaunchGeometryError(
            f"manifest spark_tp_degree={manifest.get('spark_tp_degree')!r} does "
            f"not match requested degree {degree}"
        )
    if geometry.get("intermediate") != intermediate:
        raise LaunchGeometryError(
            f"manifest intermediate={geometry.get('intermediate')!r} does not "
            f"match requested intermediate {intermediate}"
        )
    kernel_intermediate = geometry.get("kernel_intermediate")
    hidden = geometry.get("hidden")
    if kernel_intermediate is None or hidden is None:
        raise LaunchGeometryError(
            "manifest geometry must declare hidden and kernel_intermediate; "
            "refusing to guess the pinned kernel contract"
        )
    if revision is None:
        revision = manifest.get("sparkinfer_revision")
    resolved: dict[int, dict[str, Any]] = {}
    variants = manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise LaunchGeometryError("manifest declares no capacity variants")
    for index, variant in enumerate(variants):
        capacity = variant.get("capacity_rows")
        width = variant.get("width")
        output_kind = variant.get("output_kind")
        if capacity is None:
            raise LaunchGeometryError(
                f"manifest variants[{index}] is missing capacity_rows")
        if width is None:
            raise LaunchGeometryError(
                f"manifest capacity {capacity} is missing its compiled width")
        expected = launch_geometry(
            capacity,
            width,
            intermediate=intermediate,
            kernel_intermediate=kernel_intermediate,
            output_kind=output_kind,
            hidden=hidden,
            revision=revision,
        )
        recorded = variant.get("launch_geometry")
        if recorded is not None and recorded != expected:
            raise LaunchGeometryError(
                f"manifest capacity {capacity} records a launch_geometry that "
                "disagrees with the resolved pinned-kernel identity"
            )
        resolved[int(capacity)] = expected
    if not resolved:
        raise LaunchGeometryError("manifest declares no usable capacity variants")
    return resolved
