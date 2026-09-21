#!/usr/bin/env python3
"""Compare direct or packed native EXL3 AOT with B12x on real checkpoint tiles."""
from __future__ import annotations
import argparse
import ctypes as ct
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import _pinned_sparkinfer


def coverage_gate_passed(dspark_stage, uniform_global, mixed):
    """Whether a run demonstrates that the encoder reads per-projection tier membership.

    Kept separate from the family pairing in `checkpoint_global_family`: that equality
    is mandatory on every path including sampled draft stages, while this coverage
    evidence is what a draft stage legitimately cannot provide. Coverage is shown
    either by the checkpoint being globally single-width (the empty adjacent tier is
    then the whole point, and the loader knows it from the manifest rather than from
    a sample) or by some sampled expert mixing widths across its own projections.
    """
    if dspark_stage is not None:
        return True
    return bool(uniform_global or mixed)


def _load_v41_exl3_family():
    """Import the shared pure family module by sibling path (works under spec loads)."""
    import importlib.util
    import sys
    name = 'ds41rt_v41_exl3_family'
    if existing := sys.modules.get(name):
        return existing
    path = Path(__file__).resolve().with_name('v41_exl3_family.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # canonical single instance, registered pre-exec
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # A failed import must leave no half-initialised module behind, but
        # only OUR registration is ever removed — a load that already
        # succeeded (or replaced the entry meanwhile) keeps serving, and a
        # successful exec is never popped.
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


_V41_EXL3_FAMILY = _load_v41_exl3_family()
# Single source of truth shared with bench_v41_exl3_tiles: these names stay
# module-level here so the oracle and its tests keep consulting one copy of the
# loader rule (extracted verbatim; the Rust tripwire test guards drift).
expected_decoder_family = _V41_EXL3_FAMILY.expected_decoder_family
checkpoint_global_family = _V41_EXL3_FAMILY.checkpoint_global_family
_PROJECTION_NAME = _V41_EXL3_FAMILY._PROJECTION_NAME
_staged_tensor_storage = _V41_EXL3_FAMILY._staged_tensor_storage


def checkpoint_part_families(snapshot):
    """Width distribution per population: target `layers.*` vs draft `mtp.*`.

    The loader unifies both into one `decoder_family`, so the global pairing stays
    authoritative. The split matters for a different reason: the oracle only ever
    samples ONE part (`--layer` reads target, `--dspark-stage` reads mtp), and a part
    can be uniform inside a mixed checkpoint. Whether the sampled part is uniform is
    what decides if an empty tier is legitimate, so reporting both parts keeps that
    judgement on data instead of on the checkpoint-wide flag.

    Returns `{'target': {...}, 'mtp': {...}}`, each `family` / `count` / `histogram` /
    `uniform`, or `None` for a part the publication does not carry (a raw single-width
    model has no per-tensor table at all).
    """
    storage = _staged_tensor_storage(snapshot)
    if storage is None:
        return {'target': None, 'mtp': None}
    parts = {'target': [], 'mtp': []}
    unmatched = []
    for name, entry in storage.items():
        match = _PROJECTION_NAME.match(name)
        if not match:
            unmatched.append(name)
            continue
        parts['target' if match.group(1) == 'layers' else 'mtp'].append(
            (int(match.group(2)), int(match.group(3)), match.group(4),
             entry['bits_per_weight']))
    if unmatched:
        raise ValueError(
            f'{len(unmatched)} tensor_storage entries are not routed projections the '
            f'oracle can sample, for example {unmatched[:3]}')
    result = {}
    for part, rows in (('target', parts['target']), ('mtp', parts['mtp'])):
        if not rows:
            result[part] = None
            continue
        widths = [width for _, _, _, width in rows]
        result[part] = {
            'family': expected_decoder_family(widths),
            'count': len(rows),
            'experts': len({expert for _, expert, _, _ in rows}),
            'layers': sorted({layer for layer, _, _, _ in rows}),
            'histogram': {str(width): widths.count(width) for width in sorted(set(widths))},
            'uniform': len(set(widths)) == 1,
        }
    return result


def sample_expert_bound(parts, sampled_part, raw_population=None):
    """Exclusive expert-ID ceiling for the population actually being sampled.

    The 384 target experts and the (typically 128) experts of one mtp block are
    different populations: validating mtp sample indices against 384 would let
    a legal-looking request sail into a KeyError on a nonexistent tensor name.
    A raw publication without a per-tensor table is bounded by `raw_population`
    — the config-declared `n_routed_experts` — NEVER by the probe module's own
    expert count: the AOT probe reuses a handful of slots (e.g. 6), which is a
    sample size, not a population size.
    """
    part = parts.get(sampled_part)
    if part is not None:
        return int(part['experts'])
    if raw_population is None:
        raise ValueError(
            f'cannot bound --sample-experts for the sampled {sampled_part} '
            'population: the publication has no per-tensor expert table and '
            'config.json declares no n_routed_experts to fall back on')
    raw_population = int(raw_population)
    if raw_population <= 0:
        raise ValueError(
            f'config n_routed_experts must be positive, got {raw_population}')
    return raw_population


def sampled_part_policy(parts, sampled_part, uniform_global):
    """Whether the sample must exercise every declared tier, per probed population.

    An empty tier is only legitimate when the population actually being read is
    uniform, so the decision comes from that part rather than from the checkpoint as
    a whole: a mixed mtp block still has to run both kernels even if the target layers
    happen to be single-width, and vice versa. Where a publication declares no
    per-tensor table for the part at all, the checkpoint-wide flag is the only
    evidence available, except for a draft-stage run -- sampling mtp is meaningless
    if the manifest has no mtp projections, so that refuses instead of silently
    qualifying target layers.
    """
    if sampled_part not in ('target', 'mtp'):
        raise ValueError(f'unknown sampled part: {sampled_part}')
    part = parts.get(sampled_part)
    if part is None:
        if sampled_part == 'mtp':
            raise ValueError(
                '--dspark-stage samples mtp projections but this publication declares '
                'none; refusing to fall back to target layers')
        return not uniform_global, sampled_part, 'checkpoint-wide (part not declared)'
    return not part['uniform'], sampled_part, f'{sampled_part} part'


def assert_part_widths(part, name, declared):
    """Each population must be encodable by the declared tiers, no more, no less.

    Deliberately NOT family equality: a uniform mtp block inside a mixed checkpoint is
    real, and the loader's single family spans both parts.
    """
    if part is None:
        return
    widths = {int(key) for key in part['histogram']}
    if not widths <= set(declared):
        raise ValueError(
            f'{name} projections contain widths {sorted(widths - set(declared))} that the '
            f'export declares no tier for: {list(declared)} (read {part["histogram"]} '
            f'over {part["count"]} projections)')


def assert_declared_family(declared, family, projection_count, histogram):
    """Pair the export's declared tiers with the checkpoint-wide family.

    Exact, order-sensitive equality: the loader sorts `decoder_family`, the runtime
    binary-searches it, and the package directory tag is `exl3-k<digits>`, so `[4, 3]`
    names the same set but can never be served.
    """
    if list(declared) != list(family):
        raise ValueError(
            f'export declares tiers {list(declared)} but this checkpoint is globally '
            f'{histogram or "single-width"} over {projection_count} projections, which '
            f'forms the serving family {list(family)} that the loader will build')


def sample_projection_contract(declared, bitmaps, experts, projections=('w1', 'w3', 'w2'),
                               require_all_tiers=True):
    """Map which sampled projection belongs to which declared tier.

    Deliberately NOT the family check: only `checkpoint_global_family` may pair a
    declaration, because the loader's `decoder_family` runs over every projection of
    every layer while this oracle reads a handful. What the sample must still satisfy
    is narrower and load-bearing: every width it sees has to have a declared tier, and
    every declared tier has to actually be exercised, or the export is lying about the
    tensors it was compiled for.

    Returns `(members, observed, mixed)` where `members[bits][projection]` lists the
    sampled experts carrying that width, and `mixed` keeps the pinned coverage gate's
    meaning: some expert mixing widths across its own projections, which is the
    encoder's per-projection lookup.
    """
    missing = [(expert, proj) for expert in range(experts) for proj in projections
               if (expert, proj) not in bitmaps]
    if missing:
        raise ValueError(f'checkpoint is missing trellis width metadata for {missing[:4]} '
                         f'({len(missing)} entries)')
    observed = {bitmaps[expert, proj] for expert in range(experts) for proj in projections}
    if not observed:
        raise ValueError('sampled projections contributed no trellis widths to check')
    if not observed <= set(declared):
        raise ValueError(
            f'sampled projections contain widths {sorted(observed - set(declared))} that '
            f'the export declares no tier for: {list(declared)}')
    if require_all_tiers and not set(declared) <= observed:
        # Every declared tier must be numerically exercised, otherwise "the sample
        # spans the family" is a claim about the manifest, not about compiled kernels.
        # A globally uniform checkpoint is the one legitimate exception: the loader
        # retains an empty adjacent tier there, so the sample cannot show it.
        raise ValueError(
            f'export declares tiers {list(declared)} but the sampled projections only '
            f'contain {sorted(observed)}; tier {sorted(set(declared) - observed)} would '
            f'never be exercised. Select --sample-experts that cover both tiers.')
    members = {bits: {proj: tuple(e for e in range(experts) if bitmaps[e, proj] == bits)
                      for proj in projections} for bits in declared}
    mixed = any(len({bitmaps[e, proj] for proj in projections}) > 1 for e in range(experts))
    return members, observed, mixed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aot', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    layer = parser.add_mutually_exclusive_group()
    layer.add_argument('--dspark-stage', type=int, choices=(0,1,2))
    layer.add_argument('--layer', type=int, choices=range(40), default=0)
    parser.add_argument('--slice-start', type=int, default=0)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--fixture-input', type=Path, help='Reuse a prior fixture input prefix for a different capacity/tile reference')
    parser.add_argument('--fixture-canonical-routes', action='store_true',
        help='Emit valid distinct routes, using zero weights instead of masked IDs')
    parser.add_argument('--fixture-format', choices=('bf16','fp8_k32'), default='bf16')
    parser.add_argument('--sample-experts',
                        help='Comma-separated checkpoint expert indices filling the probe '
                             'slots, as many as the export has experts. Default is the '
                             'contiguous 0..n-1 range. Choose them from the checkpoint '
                             'manifest so every declared tier and any per-projection mix '
                             'is actually exercised; the choice is recorded in the report.')
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from b12x.moe.fused_moe.trellis import ProjectionTrellisTierWeights, prepare_projection_native_trellis_weights
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        compile_mixed_trellis, make_mixed_trellis_buffers, bind_mixed_trellis, run_bound_mixed_trellis,
    )
    meta = json.loads((args.aot/'v41_exl3.json').read_text())
    # The declared pair must equal the checkpoint-wide family; enforced by
    # checkpoint_global_family below, before any tensor is read.
    if len(meta['bits']) != 2:
        raise ValueError(f'this oracle qualifies a two-tier export, read tiers {meta["bits"]}')
    if meta['experts'] != 6:
        raise ValueError('this oracle qualifies a six-expert probe, not the full production '
                         f'inventory; read experts {meta["experts"]}')
    assert meta['sparkinfer_revision'] == _pinned_sparkinfer.REVISION
    paired = meta.get('paired_boundary')
    assert paired in (None, 'first', 'last')
    assert paired is None or meta.get('descriptor_rows') == 4
    hidden, width, experts, capacity = meta['hidden'], meta['intermediate'], meta['experts'], meta['capacity']
    start=args.slice_start
    assert start >= 0 and start % 128 == 0 and start+width <= 2304
    topk = meta['top_k']
    graph_rows = min(3, capacity)
    assert topk == (3 if args.dspark_stage is not None else 6)
    layer_prefix = f'mtp.{args.dspark_stage}' if args.dspark_stage is not None else f'layers.{args.layer}'
    # Authoritative pairing check, before any tensor is read or any kernel compiled:
    # the loader's family is a checkpoint-wide property, so the manifest decides it.
    global_family, global_projections, uniform_global, global_histogram = \
        checkpoint_global_family(args.snapshot)
    assert_declared_family(meta['bits'], global_family, global_projections, global_histogram)
    parts = checkpoint_part_families(args.snapshot)
    for part_name in ('target', 'mtp'):
        assert_part_widths(parts[part_name], part_name, meta['bits'])
    # Whether an empty tier is legitimate is a property of the population actually
    # being sampled, not of the checkpoint as a whole: `--layer` reads target layers,
    # `--dspark-stage` reads one mtp block. A mixed mtp part must still exercise both
    # kernels even when the target part happens to be uniform.
    require_all_tiers, sampled_part, uniformity_basis = sampled_part_policy(
        parts, 'mtp' if args.dspark_stage is not None else 'target', uniform_global)
    # Which checkpoint experts fill the probe slots. Default is contiguous 0..n so an
    # unchanged invocation behaves exactly as before; a deliberate list is recorded in
    # the report, because "which experts were sampled" is part of the evidence.
    sample_experts = (tuple(range(experts)) if args.sample_experts is None else
                      tuple(int(value) for value in args.sample_experts.split(',')))
    if len(set(sample_experts)) != len(sample_experts) or len(sample_experts) != experts:
        raise ValueError(
            f'--sample-experts needs exactly {experts} distinct expert indices, got '
            f'{sample_experts}')
    raw_population = None
    if parts.get(sampled_part) is None:
        # Raw publication (no per-tensor table): bound the sample by the
        # config-declared target population, not by the probe module's slot
        # count.  Absent or malformed config fails clean in the bound helper.
        try:
            raw_population = json.loads(
                (args.snapshot / 'config.json').read_text()).get('n_routed_experts')
        except FileNotFoundError:
            raw_population = None
    ceiling = sample_expert_bound(parts, sampled_part, raw_population)
    if any(not 0 <= expert < ceiling for expert in sample_experts):
        raise ValueError(
            f'expert indices must lie in 0..{ceiling - 1} for the sampled '
            f'{sampled_part} population ({ceiling} experts), got {sample_experts}')
    index = json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    tensors = {}; bitmaps = {}
    for slot, expert in enumerate(sample_experts):
        for projection in ['w1', 'w3', 'w2']:
            prefix = f'{layer_prefix}.ffn.experts.{expert}.{projection}'
            for suffix in ['trellis','suh','svh','mcg']:
                name = prefix+'.'+suffix
                with safe_open(args.snapshot/index[name], framework='pt', device='cpu') as f:
                    value = f.get_tensor(name)
                    if suffix == 'mcg':
                        assert int(value.item()) & 0xffffffff == 0xCBAC1FED
                        continue
                    if suffix == 'trellis':
                        bitmaps[slot, projection] = value.shape[-1] // 16
                        value = value[start//16:(start+width)//16] if projection == 'w2' else value[:, start//16:(start+width)//16]
                    elif (projection == 'w2' and suffix == 'suh') or (projection != 'w2' and suffix == 'svh'):
                        value = value[start:start+width]
                    tensors[slot, projection, suffix] = value.contiguous().to('cuda')
    def rotations(proj, suffix):
        return torch.stack([tensors[e,proj,suffix] for e in range(experts)])
    members_by_bits, sample_observed, mixed_checkpoint = sample_projection_contract(
        meta['bits'], bitmaps, experts, require_all_tiers=require_all_tiers)
    tiers=[]
    for bits in meta['bits']:
        members = members_by_bits[bits]
        def packed(proj):
            shape = (width//16, hidden//16, 16*bits) if proj=='w2' else (hidden//16, width//16, 16*bits)
            return torch.stack([tensors[e,proj,'trellis'] for e in members[proj]]) if members[proj] else torch.empty((0,*shape),dtype=torch.int16,device='cuda')
        tiers.append(ProjectionTrellisTierWeights(bits, torch.cat((packed('w1'),packed('w3'))),
            packed('w2'), members['w1'],members['w3'],members['w2']))
    prepared = prepare_projection_native_trellis_weights(tuple(tiers),
        gate_suh=rotations('w1','suh'), up_suh=rotations('w3','suh'),
        intermediate_rotations=torch.cat((rotations('w1','svh'),rotations('w3','svh'),rotations('w2','suh')),dim=1),
        down_svh=rotations('w2','svh'), activation='silu',params_dtype=torch.bfloat16,
        num_experts=experts, hidden_size=hidden, intermediate_size=width)
    # Empty projection membership is legal. Keep one unreachable physical
    # plane for the native binding's non-null storage ABI; descriptor membership
    # and gate/up counts remain unchanged, so this adds no routed expert.
    padded=[]
    for bits,tier in zip(meta['bits'],prepared.tiers):
        stride=(width//16)*(hidden//16)*(8*bits)
        updates={}
        if tier.w13.numel()==0:
            updates['w13']=torch.zeros(stride,dtype=torch.int32,device='cuda')
        if tier.w2.numel()==0:
            updates.update(w2=torch.zeros(stride,dtype=torch.int32,device='cuda'),
                w2_global_scale=torch.ones(1,dtype=torch.float32,device='cuda'))
        padded.append(replace(tier,**updates))
    prepared=replace(prepared,tiers=tuple(padded))
    props=torch.cuda.get_device_properties(0)
    launch=compile_mixed_trellis(size_m=capacity,hidden_size=hidden,intermediate_size=width,
        tier0_num_experts=experts,tier1_num_experts=experts,route_num_experts=experts,
        top_k=topk,max_m_blocks=meta['route_blocks'],sms=props.multi_processor_count,
        max_shared_mem=props.shared_memory_per_block_optin,force_tile_config=tuple(meta['tile']),
        tier0_bits=meta['bits'][0], tier1_bits=meta['bits'][1],
        swiglu_limit=10.0, direct_topk_routes=meta['direct'],full_rotation_output_dtype=meta['output_dtype'],
        force_blocks_per_sm=meta['blocks_per_sm'] if meta['blocks_per_sm'] > 1 else None,
        **({'paired_boundary':paired} if paired is not None else {}))
    # B12x derives every tier extent from the LAUNCH, not from the declared tier
    # objects, so a reference compiled at the wrong widths fails later as an opaque
    # storage-extent error. Assert the compiled pair instead of trusting defaults.
    assert (int(launch.tier0_bits), int(launch.tier1_bits)) == (int(meta['bits'][0]), int(meta['bits'][1])), \
        (launch.tier0_bits, launch.tier1_bits, meta['bits'])
    buffers=make_mixed_trellis_buffers(launch,device=torch.device('cuda',0),sms=props.multi_processor_count)
    # Native route initialization covers the packer's rounded bucket. Use the
    # exported allocation contract for both paths, including non-power-of-two
    # capacities where it is larger than B12x's exact-capacity metadata arrays.
    buffers=replace(buffers, **{name:torch.empty(meta['buffers'][name]['shape'],
        device='cuda',dtype=torch.int32)
        for name in ('packed_route_indices','block_expert_ids')})
    descriptor=prepared.descriptor_map
    ownership=None
    if paired is not None:
        descriptor=torch.cat((descriptor,torch.ones(experts*2,device='cuda',dtype=torch.int32)))
        descriptor._mt_projection_counts=prepared.descriptor_map._mt_projection_counts
        ownership=descriptor[-experts*2:]
    binding=bind_mixed_trellis(*prepared.tiers,prepared.global_to_combined,descriptor,prepared.rotations,launch,
        gate_experts=prepared.gate_counts,up_experts=prepared.up_counts)
    lib=ct.CDLL(str(args.aot/'libds41rt_exl3.so'))
    info_verified=False
    if hasattr(lib,'ds41rt_exl3_info'):
        lib.ds41rt_exl3_info.argtypes=[ct.POINTER(ct.c_uint32),ct.c_uint32]
        lib.ds41rt_exl3_info.restype=ct.c_int
        if paired is not None:
            assert lib.ds41rt_exl3_info((ct.c_uint32*16)(),16)!=0
            query=lib.ds41rt_exl3_paired_info
            query.argtypes=[ct.POINTER(ct.c_uint32),ct.c_uint32];query.restype=ct.c_int
            count=18
        else:
            query=lib.ds41rt_exl3_info;count=16
        native_info=(ct.c_uint32*count)()
        assert query(native_info,count)==0
        expected_info=[3 if paired else 2,hidden,width,experts,capacity,topk,2,
            *[len(e[key]) for e in meta['objects'] for key in ['pointer_slots','scalar_slots']], *meta['bits'], 0, 0,
            2 if meta['output_dtype']=='bf16' else 4]
        if paired: expected_info += [1 if paired=='first' else 2,4]
        assert list(native_info)==expected_info
        info_verified=True
    lib.ds41rt_exl3_create.argtypes=[ct.POINTER(ct.c_void_p)];lib.ds41rt_exl3_create.restype=ct.c_int
    lib.ds41rt_exl3_destroy.argtypes=[ct.c_void_p]
    for role in ['core','sum']:
        fn=getattr(lib,'ds41rt_exl3_'+role)
        fn.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_int32),ct.c_void_p];fn.restype=ct.c_int
    context=ct.c_void_p();assert lib.ds41rt_exl3_create(ct.byref(context))==0
    torch.manual_seed(4105)
    x=torch.randn(capacity,hidden,device='cuda',dtype=torch.bfloat16)
    ids=(torch.arange(capacity*topk,device='cuda',dtype=torch.int32).reshape(capacity,topk) % experts)
    weights=torch.softmax(torch.randn(capacity,topk,device='cuda'),dim=1)
    pointers={name:getattr(buffers,name) for name in meta['buffers']}
    pointers.update(rotation_input_ptr=x,raw_topk_ids=ids,topk_weights_ptr=weights,
        descriptor_map_ptr=binding.descriptor_map,global_to_combined_ptr=binding.global_to_combined,
        intermediate_rotations_ptr=binding.rotations.intermediate,gate_suh_ptr=binding.rotations.gate_suh,
        up_suh_ptr=binding.rotations.up_suh,trellis_lut_ptr=launch.trellis_lut,
        fc2_ptr=buffers.fc2,output_ptr=buffers.output,route_expert_ids_ptr=ids,
        expert_map_ptr=binding.global_to_combined,svh_ptr=binding.rotations.down_svh)
    scalars=dict(grid_x=meta['blocks_per_sm']*meta['sms'],route_num_experts=experts,
        weight_num_experts=launch.topk_sum.num_experts)
    for i,tier in enumerate(prepared.tiers):
        for key,field in [('w13','w13'),('w2','w2'),('w13_scales','w13_scale'),('w2_scales','w2_scale'),('w13_global','w13_global_scale'),('w2_global','w2_global_scale')]:
            pointers[f't{i}_{key}_ptr']=getattr(tier,field)
        scalars.update({f'tier{i}_num_experts':experts,f'tier{i}_fc2_experts':binding.fc2_counts[i],
            f'tier{i}_gate_experts':binding.gate_counts[i],f'tier{i}_up_experts':binding.up_counts[i]})
    route_context=ct.c_void_p(); route_lib=None
    if not meta['direct']:
        route_path=args.aot/meta['route_preparation']['manifest']
        assert hashlib.sha256(route_path.read_bytes()).hexdigest()==meta['route_preparation']['sha256']
        route_meta=json.loads(route_path.read_text())
        route_lib=ct.CDLL(str(route_path.parent/'libv41_exl3_routes.so'))
        route_lib.ds41rt_exl3_routes_create.argtypes=[ct.POINTER(ct.c_void_p)]
        route_lib.ds41rt_exl3_routes_create.restype=ct.c_int
        route_lib.ds41rt_exl3_routes_destroy.argtypes=[ct.c_void_p]
        route_lib.ds41rt_exl3_routes_destroy.restype=ct.c_int
        route_lib.ds41rt_exl3_routes_launch.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_uint64),ct.c_int32,ct.c_void_p]
        route_lib.ds41rt_exl3_routes_launch.restype=ct.c_int
        route_tensors=dict(topk_ids=ids,expert_map=binding.global_to_combined,
            **{name:getattr(buffers,name) for name in list(route_meta['buffers'])[2:]})
        route_p=(ct.c_void_p*7)(*[t.data_ptr() for t in route_tensors.values()])
        route_bytes=(ct.c_uint64*7)(*[t.numel()*t.element_size() for t in route_tensors.values()])
        assert route_lib.ds41rt_exl3_routes_create(ct.byref(route_context))==0
    def native(rows):
        scalars['active_m']=rows
        if route_lib is not None:
            assert route_lib.ds41rt_exl3_routes_launch(route_context,route_p,route_bytes,rows,
                ct.c_void_p(torch.cuda.current_stream().cuda_stream))==0
        for entry in meta['objects']:
            role=entry['label'].rsplit('_',1)[1]
            p=(ct.c_void_p*len(entry['pointer_slots']))(*(pointers[n].data_ptr() for n in entry['pointer_slots']))
            s=(ct.c_int32*len(entry['scalar_slots']))(*(scalars[n] for n in entry['scalar_slots']))
            status=getattr(lib,'ds41rt_exl3_'+role)(context,p,s,ct.c_void_p(torch.cuda.current_stream().cuda_stream))
            assert status==0,(role,status)
    def poison_metadata():
        if route_lib is not None:
            for name in ('packed_route_indices','block_expert_ids','packed_route_count','expert_offsets','expert_counts'):
                getattr(buffers,name).fill_(-777)
    # Coverage gate, distinct from the family pairing above (which applies to every
    # path, sampled draft stages included). It proves the encoder really reads
    # per-projection tier membership. A checkpoint can demonstrate that two ways:
    # some expert mixing widths across its own projections, or a genuinely uniform
    # full model, whose empty adjacent tier is exactly what the loader retains.
    if not coverage_gate_passed(args.dspark_stage, uniform_global, mixed_checkpoint):
        raise ValueError(
            'per-projection tier membership is unverified: the checkpoint is not '
            'globally single-width and no sampled expert mixes trellis widths across '
            'its own w1/w3/w2 projections. Sample more experts, use --dspark-stage for '
            'a uniformly quantized sampled draft stage, or qualify a checkpoint whose '
            'manifest is genuinely uniform.')
    results=[]; graph=None
    try:
        for rows in sorted(set([1,min(3,capacity),max(1,capacity-1),capacity])):
            expected=run_bound_mixed_trellis(x[:rows],weights[:rows],ids[:rows],binding,buffers).clone()
            buffers.output.fill_(float('nan'));poison_metadata();native(rows);torch.cuda.synchronize()
            assert torch.isfinite(expected).all() and bool(expected.abs().sum()>0)
            assert torch.equal(buffers.output[:rows],expected)
            results.append({'rows':rows,'bitwise_equal':True})
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): native(graph_rows)
        x.mul_(1.1);ids.copy_(ids.roll(1,dims=1))
        ids[::2,1]=-1
        expected=run_bound_mixed_trellis(x[:graph_rows],weights[:graph_rows],ids[:graph_rows],binding,buffers).clone()
        buffers.output.fill_(float('nan'));poison_metadata();graph.replay();torch.cuda.synchronize()
        assert torch.equal(buffers.output[:graph_rows],expected)
        ownership_checks=[]
        if ownership is not None:
            for pattern in ([0]*experts, [i%2 for i in range(experts)], [(i+1)%2 for i in range(experts)], [1]*experts):
                ownership.fill_(0)
                ownership[:experts].copy_(torch.tensor(pattern,device='cuda',dtype=torch.int32))
                x.mul_(-.75)
                expected=run_bound_mixed_trellis(x[:graph_rows],weights[:graph_rows],ids[:graph_rows],binding,buffers).clone()
                assert torch.isfinite(expected).all() and expected.abs().max()>0
                buffers.fc1.fill_(float('nan'));buffers.output.fill_(float('nan'));poison_metadata()
                graph.replay();torch.cuda.synchronize()
                assert torch.equal(buffers.output[:graph_rows],expected)
                ownership_checks.append({'owners':pattern,'rows':graph_rows,'bitwise_equal':True})
        if args.fixture is not None:
            args.fixture.mkdir(parents=True,exist_ok=True)
            if args.fixture_canonical_routes:
                ids.copy_(torch.arange(capacity*topk,device=ids.device,dtype=ids.dtype)
                    .reshape(capacity,topk).remainder(experts).roll(1,dims=1))
                weights[::2,1]=0
            fixture_input=x
            if args.fixture_input is not None:
                source_meta=json.loads((args.fixture_input/'fixture.json').read_text())
                assert source_meta['input_format']==args.fixture_format
                assert source_meta['snapshot_revision']==args.snapshot.name
                # TP ranks slice intermediate weights, not the replicated
                # hidden input or route IDs. Reuse the same inputs across ranks.
                assert source_meta['layer']==layer_prefix and source_meta['topk']==topk
                assert source_meta['capacity']>=capacity
                assert source_meta['canonical_routes']==args.fixture_canonical_routes
                def source_tensor(name,dtype,columns):
                    raw=(args.fixture_input/(name+'.bin')).read_bytes()
                    assert len(raw)==source_meta['artifacts'][name]['bytes']
                    assert hashlib.sha256(raw).hexdigest()==source_meta['artifacts'][name]['sha256']
                    return torch.frombuffer(bytearray(raw),dtype=dtype).reshape(-1,columns)[:capacity].clone().to('cuda')
                ids.copy_(source_tensor('ids',torch.int32,topk))
                weights.copy_(source_tensor('weights',torch.float32,topk))
                fixture_input=source_tensor('input',torch.uint8 if args.fixture_format=='fp8_k32' else torch.bfloat16,
                    5280 if args.fixture_format=='fp8_k32' else hidden)
                if args.fixture_format=='bf16': x=fixture_input
            if args.fixture_format=='fp8_k32':
                if args.fixture_input is None:
                    from qualify_v41_exl3_wire import quantize_wire
                    fixture_input=torch.empty(capacity,5280,dtype=torch.uint8,device='cuda')
                    quantize_wire(x,fixture_input)
                values=fixture_input[:,:5120].contiguous().view(torch.float8_e4m3fn).float()
                scales=fixture_input[:,5120:].int().repeat_interleave(32,dim=1)-127
                x=torch.ldexp(values,scales).to(torch.bfloat16)
            fixture_owners = None
            if ownership is not None:
                starts = {0: 0, 512: 1, 1152: 2, 1664: 3}
                assert start in starts, 'paired fixture must use a real TP4 resident slice'
                rank = starts[start]
                assert paired == ('last' if rank % 2 == 0 else 'first')
                fixture_owners = torch.arange(experts, device='cuda', dtype=torch.int32).remainder(4)
                if args.fixture_input is not None and 'owners' in source_meta['artifacts']:
                    raw = (args.fixture_input/'owners.bin').read_bytes()
                    spec = source_meta['artifacts']['owners']
                    assert len(raw) == spec['bytes'] and hashlib.sha256(raw).hexdigest() == spec['sha256']
                    fixture_owners = torch.frombuffer(bytearray(raw), dtype=torch.int32).clone().to('cuda')
                    assert fixture_owners.numel() == experts and ((fixture_owners >= 0) & (fixture_owners < 4)).all()
                ownership.zero_()
                ownership[:experts].copy_(((fixture_owners >> (rank // 2)) & 1) == rank % 2)
            expected=run_bound_mixed_trellis(x,weights,ids,binding,buffers).clone()
            artifacts={}
            values = [('input',fixture_input),('ids',ids),('weights',weights),('expected',expected)]
            if fixture_owners is not None: values.append(('owners', fixture_owners))
            for name,value in values:
                raw=value.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                (args.fixture/(name+'.bin')).write_bytes(raw)
                artifacts[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            (args.fixture/'fixture.json').write_text(json.dumps({'layer':layer_prefix,'slice_start':start,
                'width':width,'capacity':capacity,'topk':topk,'reference_experts':experts,
                'direct':meta['direct'],'tile':meta['tile'],
                'input_format':args.fixture_format,'output_dtype':meta['output_dtype'],
                'canonical_routes':args.fixture_canonical_routes,
                'paired_boundary':paired,'paired_rank':None if paired is None else rank,
                'input_fixture_manifest_sha256':None if args.fixture_input is None else
                    hashlib.sha256((args.fixture_input/'fixture.json').read_bytes()).hexdigest(),
                'snapshot_revision':args.snapshot.name,'artifacts':artifacts},indent=2)+'\n')
        args.output.write_text(json.dumps({'passed':True,'scope':f'native AOT versus B12x on six real checkpoint experts at tiers {meta["bits"]}; '
                    'not the 384-expert production inventory and not full-model qualification',
            'tier_family': list(meta['bits']),
            'tier_family_expected': expected_decoder_family(
                {bitmaps[e, p] for e in range(experts) for p in ['w1', 'w3', 'w2']}),
            # Two distinct evidence modes, stated explicitly so a synthetic run can
            # never be read as checkpoint evidence: --fixture feeds the compiled
            # modules a generated prefix (proves the per-projection tier lookup on
            # constructed widths, with no claim about this checkpoint), while a real
            # run samples checkpoint experts. `tier_exercise` counts sampled
            # projections per declared tier, so a zero here says a tier kernel was
            # never numerically exercised -- legitimate only for the padded empty
            # tier of a globally uniform checkpoint.
            'evidence_mode': 'synthetic-fixture' if args.fixture is not None else 'real-checkpoint',
            'tier_exercise': {str(bits): sum(len(members) for members in
                                             members_by_bits[bits].values())
                              for bits in meta['bits']},
            'sample_experts': list(sample_experts),
            'sample_observed_tiers': sorted(sample_observed),
            'tier_family_global': global_family,
            'tier_global_histogram': global_histogram,
            'tier_global_projection_count': global_projections,
            'tier_uniform_global': uniform_global,
            'sampled_part': sampled_part,
            'sampled_part_uniformity_basis': uniformity_basis,
            'require_all_tiers': require_all_tiers,
            'tier_target_family': None if parts['target'] is None else parts['target']['family'],
            'tier_target_count': None if parts['target'] is None else parts['target']['count'],
            'tier_target_histogram': None if parts['target'] is None else parts['target']['histogram'],
            'tier_target_uniform': None if parts['target'] is None else parts['target']['uniform'],
            'tier_mtp_family': None if parts['mtp'] is None else parts['mtp']['family'],
            'tier_mtp_count': None if parts['mtp'] is None else parts['mtp']['count'],
            'tier_mtp_histogram': None if parts['mtp'] is None else parts['mtp']['histogram'],
            'tier_mtp_uniform': None if parts['mtp'] is None else parts['mtp']['uniform'],
            'reference_tier_bits': [int(launch.tier0_bits), int(launch.tier1_bits)],
            'experts_qualified': experts, 'full_production_experts': 384,
            'output_dtype':meta['output_dtype'],'checkpoint_layer':layer_prefix,'topk':topk,'native_info_verified':info_verified,
            'slice_start':start,'paired_boundary':paired,'ownership_graph_checks':ownership_checks,
            'sparkinfer_revision':_pinned_sparkinfer.REVISION,'compute':meta['compute'],'intermediate':width,
            'direct':meta['direct'],
            'route_bridge_sha256':None if route_lib is None else hashlib.sha256((route_path.parent/'libv41_exl3_routes.so').read_bytes()).hexdigest(),
            'snapshot_revision':args.snapshot.name,
            'projection_tiers':[[bitmaps[e,p] for p in ['w1','w3','w2']] for e in range(experts)],
            'aot_manifest_sha256':hashlib.sha256((args.aot/'v41_exl3.json').read_bytes()).hexdigest(),
            'bridge_sha256':hashlib.sha256((args.aot/'libds41rt_exl3.so').read_bytes()).hexdigest(),
            'checks':results,'graph_changed_inputs_and_routes':True,
            'packed_metadata_poisoned_before_native':route_lib is not None},indent=2)+'\n')
        print(args.output.read_text(),flush=True)
    finally:
        torch.cuda.synchronize()
        if graph is not None: del graph
        if route_lib is not None: assert route_lib.ds41rt_exl3_routes_destroy(route_context)==0
        lib.ds41rt_exl3_destroy(context)


if __name__=='__main__': main()
