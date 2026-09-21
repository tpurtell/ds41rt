#!/usr/bin/env python3
"""Offline Spark tile/residency experiment; never changes the serving policy.

Two disjoint-package modes share one harness:

* ``--intermediate 512|640 --slice-start S`` keeps the legacy TP4 paired-width
  geometry exactly as this harness has always raced it: the production
  baseline plus the same-tile 1-vs-2 blocks/SM residency pair.  Forcing
  blocks/SM is a *paired-only* knob (see
  ``package_v41_exl3_aot.residency_overrides``), so it is legal only here.
* ``--intermediate 768 --slice-start {0,768,1536}`` benchmarks the real
  disjoint EXL3 TP3 rank slices (the ``tp3-width768`` package layouts).  A
  width of 768 is exactly six whole H128 blocks with no padding, so the only
  admissible slice starts are the three exact rank thirds of 2304.  This mode
  races the real production planner tile against controlled tile candidates
  ((64,256) vs (64,128), and the m10..m32 (128,128) geometry) with automatic
  residency only: blocks/SM is recorded as the compiler resolved it and is
  never forced, so paired residency semantics cannot leak into a disjoint run.

The tier family is never inferred from the checkpoint name, repository id, or
nominal bits-per-weight.  It is derived from the checkpoint's own per-tensor
MCG trellis widths, must equal the family the loader builds for the WHOLE
checkpoint (``checkpoint_global_family`` over the publication manifest), is
optionally declared with ``--tiers``, and must agree; any other family fails
closed before anything compiles or times.

Deliberate scope: the EXL3 package path serves the disjoint TP3 width-768 rank
slices and the legacy TP4 paired 512/640 widths, and only those are benchmark.
TP2 (width 1152) is exportable but has NO APPROVED EXL3 TP2 serving config in
this repository, so there is no sanctioned deployment whose geometry a
measurement would describe; TP6 (width 384) has no package profile at all.
Both are therefore excluded on purpose, not by omission
(``resolve_layout`` and the CLI choices reject them fail-closed).

Uses all 384 experts of a checkpoint layer and compares against the production
tile.  Graph timings include routing, mixed core and reduction, not transport.
Correctness gates (finite/nonzero, cross-variant agreement against the
production baseline, eager/graph bitwise equality, replay determinism, live
row-count schedule independence, stable preplanned workspace with zero replay
allocations) all run before any timing sample is taken.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import _pinned_sparkinfer


def _load_v41_exl3_family():
    """Import the shared pure family module by sibling path (works under spec loads)."""
    import importlib.util
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


# One copy of the loader family rule, shared with qualify_v41_exl3_aot (the
# Rust loader tripwire in test_qualify_v41_exl3_aot_tiers guards its drift).
_V41_EXL3_FAMILY = _load_v41_exl3_family()
expected_decoder_family = _V41_EXL3_FAMILY.expected_decoder_family
checkpoint_global_family = _V41_EXL3_FAMILY.checkpoint_global_family

TOTAL_INTERMEDIATE = 2304
TP3_WIDTH = 768
TP3_RANK_STARTS = (0, 768, 1536)
LEGACY_PAIRED_WIDTHS = (512, 640)
H128_BLOCK = 128
MCG_MAGIC = 0xCBAC1FED
# The packed-router block and every compile option the exported EXL3 package is
# built with (`export_b12x_v41_exl3_aot` pins block_m = 8 and passes these
# kwargs explicitly).  The bench mirrors that call exactly so the raced
# geometry is the geometry the shipped AOT module was compiled under.
MOE_BLOCK_SIZE = 8
COMPILE_PARITY_KWARGS = dict(trellis_codebook='mcg', moe_block_size=MOE_BLOCK_SIZE,
                             rotation_input_dtype='bf16',
                             full_rotation_output_dtype='bf16', swiglu_limit=10.)
CHECKPOINT_RE = re.compile(r'^[\w.-]+/[\w.-]+$')
# Substrings of the PINNED B12x compiler's own tile/residency/resource
# rejections.  A controlled candidate that trips one of these is reported
# illegal for this geometry; any other failure aborts the run.  These markers
# are scoped to the single sparkinfer revision this harness pins (recorded as
# source.sparkinfer_revision): on any pin bump they must be re-audited against
# that revision's error strings, since a renamed message could silently turn a
# declined candidate into an abort (fail-closed, safe) or, worse, a real error
# into a skipped candidate (never intended).  The pinned planner stays the sole
# authority on tile legality: this list only maps ValueError text to "the
# compiler declined this candidate", it never encodes which geometries are
# legal.
TILE_LEGALITY_MARKERS = (
    'divisible by tile',
    'multiples of 16',
    'thread count',
    'cta threads',
    'shared-memory footprint',
    'shared-memory requirement exceeds',
    'residency',
    'tile_k that',
    'tile_n=256',
)


def resolve_layout(intermediate, slice_start):
    """Validate one offline slice request; return its disjoint/paired layout record."""
    intermediate = int(intermediate)
    slice_start = int(slice_start)
    if intermediate not in (*LEGACY_PAIRED_WIDTHS, TP3_WIDTH):
        raise ValueError(f'unsupported benchmark width {intermediate}')
    if slice_start < 0 or slice_start + intermediate > TOTAL_INTERMEDIATE:
        raise ValueError(
            f'slice [{slice_start}, {slice_start + intermediate}) leaves the '
            f'{TOTAL_INTERMEDIATE}-wide intermediate')
    if intermediate == TP3_WIDTH:
        # Disjoint TP3 is exactly three unpadded thirds of 2304, each six whole
        # H128 blocks. Anything else is a padded/paired request wearing a 768.
        if slice_start not in TP3_RANK_STARTS:
            raise ValueError(
                f'disjoint TP3 width {TP3_WIDTH} admits slice starts '
                f'{TP3_RANK_STARTS} only, got {slice_start}')
        return dict(mode='tp3-disjoint-width768', intermediate=intermediate,
                    slice_start=slice_start, rank=slice_start // TP3_WIDTH,
                    whole_h128_blocks=intermediate // H128_BLOCK, padded=False)
    if slice_start % H128_BLOCK != 0:
        raise ValueError('legacy TP4 slices must start on an H128 block boundary')
    return dict(mode='tp4-paired-legacy', intermediate=intermediate,
                slice_start=slice_start, rank=None,
                whole_h128_blocks=intermediate // H128_BLOCK, padded=True)


def parse_tiers(spec):
    """`--tiers 3,4` -> (3, 4); None/''/'auto' -> None (derive from checkpoint)."""
    if spec is None or str(spec).strip().lower() in ('', 'auto'):
        return None
    try:
        tiers = tuple(int(part) for part in str(spec).split(','))
    except ValueError:
        raise ValueError(f'--tiers must be two integers like 3,4, got {spec!r}') from None
    if len(tiers) != 2 or len(set(tiers)) != 2:
        raise ValueError(f'--tiers must declare exactly two distinct tier widths, got {spec!r}')
    if tiers != tuple(sorted(tiers)):
        raise ValueError(f'--tiers must be ascending (tier0, tier1), got {spec!r}')
    return tiers


def resolve_tier_family(declared, observed, global_family=None, global_audit=None):
    """Bind an explicit or derived two-tier family to the checkpoint's widths.

    The pairing rule itself is the shared `v41_exl3_family.expected_decoder_family`
    (the Rust loader mirror; widths must be plain non-bool ints).  Fails closed
    on: non-K2..K5 widths, a declared family that is not the one the checkpoint's
    real widths form, and any family this two-tier mixed kernel path cannot
    compile (three or more distinct widths).  When the caller supplies the
    checkpoint-GLOBAL family (from `checkpoint_global_family`, i.e. every
    projection of every layer of the publication manifest), the family derived
    from the sampled layer must equal it: one layer may never legitimise a
    family the loader would not build for the whole checkpoint.
    """
    observed = list(observed)
    expected = expected_decoder_family(observed)
    if len(expected) > 2:
        raise ValueError(
            'this two-tier offline harness cannot run a mixed checkpoint family with '
            f'{len(expected)} widths {expected}; the disjoint TP3 package is two-tier')
    record = dict(observed_widths=sorted(set(observed)),
                  sampled_layer_family=list(expected),
                  derived_family=list(expected), declared=None if declared is None
                  else [int(value) for value in declared],
                  rule='shared v41_exl3_family.expected_decoder_family mirroring '
                       'rust/crates/ds41rt-loader/src/v41_exl3.rs decoder_family')
    if global_family is not None:
        global_family = list(global_family)
        if global_family != expected:
            raise ValueError(
                f'the sampled layer forms family {expected}, but the checkpoint-'
                f'global widths form {global_family}; refusing to benchmark a '
                'family the loader would not build for this checkpoint')
        record['checkpoint_global_family'] = global_family
        if global_audit is not None:
            record['checkpoint_global'] = dict(global_audit)
    if declared is None:
        record['resolution'] = ('derived-from-checkpoint-trellis-widths'
                                if global_family is None else
                                'derived-from-checkpoint-trellis-widths-matching-global')
    else:
        declared = tuple(int(value) for value in declared)
        if list(declared) != expected:
            raise ValueError(
                f'declared tiers {list(declared)} are not the family the checkpoint '
                f'widths form ({expected}); refusing to benchmark a mismatched tier set')
        record['resolution'] = ('declared-and-confirmed-by-checkpoint-trellis-widths'
                                if global_family is None else
                                'declared-and-confirmed-matching-checkpoint-global')
    record['tiers'] = list(expected)
    return tuple(expected), record


def plan_variants(intermediate, planner_tile):
    """Enumerate the raced variants for one mode as pure data.

    TP3 (disjoint width 768): the production planner tile plus the controlled
    candidates (64,256,64,256), (64,128,64,128), (128,128,128,128), never a
    forced blocks/SM.  A candidate equal to the planner tile is a duplicate of
    the production geometry (that is the m16 case: the planner itself picks
    (128,128,128,128)), so it is recorded but not recompiled.
    Legacy widths: the historical race, byte-for-byte — planner tile with the
    historical forced-1 baseline, plus the same-tile (64,128) 1-vs-2
    blocks/SM paired-residency pair.
    """
    planner_tile = tuple(int(value) for value in planner_tile)
    if intermediate == TP3_WIDTH:
        specs = [dict(name='baseline', kind='production-planner',
                      tile=planner_tile, blocks_per_sm_forced=None)]
        for name, tile in (('k64-n256', (64, 256, 64, 256)),
                           ('k64-n128', (64, 128, 64, 128)),
                           ('k128-n128', (128, 128, 128, 128))):
            spec = dict(name=name, kind='controlled-tile', tile=tile,
                        blocks_per_sm_forced=None)
            if tile == planner_tile:
                spec['skip'] = 'duplicate-of-production-planner'
            specs.append(spec)
        return specs
    return [dict(name='baseline', kind='production-planner', tile=planner_tile,
                 blocks_per_sm_forced=1),
            dict(name='k64-one-block', kind='paired-residency', tile=(64, 128, 64, 128),
                 blocks_per_sm_forced=1),
            dict(name='k64-two-blocks', kind='paired-residency', tile=(64, 128, 64, 128),
                 blocks_per_sm_forced=2)]


def is_tile_legality_error(message):
    lowered = str(message).lower()
    return any(marker in lowered for marker in TILE_LEGALITY_MARKERS)


STAGED_MANIFEST_SCHEMA = 'ds41rt-hf-staged-snapshot-v1'
# A staged revision is either a git commit (40-hex, e.g. the HF snapshot
# directory name) or a content-addressed manifest hash (64-hex, the
# sync_ds4_hf_snapshot contract).  Both are admitted; the manifest must echo
# whichever one names the `snapshots/<revision>` directory.
REVISION_RE = re.compile(r'^(?:[0-9a-f]{40}|[0-9a-f]{64})$')


def checkpoint_identity(snapshot, expected_checkpoint=None):
    """Bind the run to explicit checkpoint identity; never to a name guess.

    Records the resolved path, staged revision directory, and the SHA-256 of
    the weight index.  When a ``ds41rt-manifests/<revision>.json`` is present
    the snapshot is verified against the same fields `sync_ds4_hf_snapshot`
    checks: exact ``schema``, exact ``model_id``, and a manifest revision
    (``revision``, or its ``commit`` field where a publisher used that name)
    equal to the ``snapshots/<revision>`` directory, plus a NONEMPTY ``files``
    list (an empty table certifies nothing).
    The sync helper's canonical ``sha256({"schema","files"})`` is recorded and
    only enforced when it is the revision scheme in use (a 64-hex manifest that
    hashes to itself); a git-commit revision is legitimately not that hash, so
    there the manifest is accepted on its explicit model_id/revision equality.
    Any field mismatch fails closed.  A staged cache root that carries a
    ``refs/main`` pointer must name exactly the revision under test
    (``refs_main_matches``): a stale default-checkout ref would otherwise let a
    run benchmark a snapshot the cache no longer considers current.  Without a
    manifest, an ``--checkpoint``
    id is only confirmable through the exact ``models--org--repo/snapshots``
    cache structure.  Nominal ``quantize_config.json`` fields (bits/format/
    codebook) are recorded as audit data only and never feed the tier decision.
    """
    snapshot = Path(snapshot)
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise ValueError(f'checkpoint snapshot must be a real directory, not a symlink: {snapshot}')
    resolved = snapshot.resolve()
    index = resolved / 'model.safetensors.index.json'
    if not index.is_file():
        raise ValueError(f'checkpoint snapshot has no model.safetensors.index.json: {resolved}')
    staged = resolved.parent.name == 'snapshots' and bool(REVISION_RE.match(resolved.name))
    revision = resolved.name if staged else None
    index_sha256 = hashlib.sha256(index.read_bytes()).hexdigest()
    record = dict(snapshot=str(resolved), revision=revision,
                  index_sha256=index_sha256, requested=None, confirmed_via=None,
                  manifest_model_id=None, manifest_verified=False,
                  manifest_revision_scheme=None, manifest_content_sha256=None,
                  refs_main_matches=None,
                  nominal_quantize_metadata=None)
    manifest_model_id = None
    if revision is not None:
        # A staged HF cache tracks its default checkout in refs/main; if that
        # pointer exists it must name THIS snapshot, or the run would
        # benchmark a revision the cache no longer considers current.
        refs_main = resolved.parent.parent / 'refs' / 'main'
        if refs_main.is_file():
            pointer = refs_main.read_text().strip()
            if pointer != revision:
                raise ValueError(
                    f'cache refs/main points at {pointer!r}, not the snapshot '
                    f'revision {revision!r} under test; refusing to benchmark a '
                    'stale ref')
            record['refs_main_matches'] = True
        manifest = resolved.parent.parent / 'ds41rt-manifests' / f'{revision}.json'
        if manifest.is_file():
            payload = json.loads(manifest.read_text())
            if payload.get('schema') != STAGED_MANIFEST_SCHEMA:
                raise ValueError(
                    f'staged manifest {manifest} has schema {payload.get("schema")!r}, '
                    f'expected {STAGED_MANIFEST_SCHEMA!r}')
            manifest_model_id = payload.get('model_id')
            manifest_revision = payload.get('revision', payload.get('commit'))
            if manifest_revision != revision:
                raise ValueError(
                    f'staged manifest revision {manifest_revision!r} does not match the '
                    f'snapshots/{revision} directory it certifies')
            files = payload.get('files')
            if not isinstance(files, list) or not files:
                raise ValueError(
                    f'staged manifest {manifest} carries no nonempty files list')
            content_sha = hashlib.sha256(json.dumps(
                {'schema': STAGED_MANIFEST_SCHEMA, 'files': files},
                sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            record['manifest_content_sha256'] = content_sha
            if len(revision) == 64:
                # Content-addressed scheme: the sync helper requires the manifest
                # to hash to the revision directory name; honor that exactly.
                if content_sha != revision:
                    raise ValueError(
                        f'staged manifest {manifest} does not hash to the revision '
                        f'directory name it certifies (sync helper contract)')
                record['manifest_revision_scheme'] = 'content-sha256'
            else:
                # Git-commit scheme: revision is a source commit, legitimately
                # not the manifest content hash.  Identity rests on the exact
                # schema/model_id/revision equality verified above.
                record['manifest_revision_scheme'] = 'git-commit'
            if not isinstance(manifest_model_id, str) or not manifest_model_id:
                raise ValueError(f'staged EXL3 manifest has no model identity: {manifest}')
            record['manifest_verified'] = True
    record['manifest_model_id'] = manifest_model_id
    quantize = resolved / 'quantize_config.json'
    if quantize.is_file():
        try:
            meta = json.loads(quantize.read_text())
        except json.JSONDecodeError:
            meta = {}
        record['nominal_quantize_metadata'] = {
            key: meta.get(key) for key in ('bits', 'checkpoint_format', 'codebook')}
    if expected_checkpoint is None:
        return record
    expected_checkpoint = str(expected_checkpoint)
    if not CHECKPOINT_RE.fullmatch(expected_checkpoint):
        raise ValueError(f'--checkpoint must be an org/repo id, got {expected_checkpoint!r}')
    record['requested'] = expected_checkpoint
    if manifest_model_id is not None:
        if manifest_model_id != expected_checkpoint:
            raise ValueError(
                f'staged snapshot is {manifest_model_id!r}, not the requested '
                f'checkpoint {expected_checkpoint!r}')
        record['confirmed_via'] = 'ds41rt-manifest-verified'
    else:
        slug = 'models--' + expected_checkpoint.replace('/', '--')
        # No manifest to verify against: only the exact staged cache structure
        # (…/models--org--repo/snapshots/<revision>) may confirm the id.
        if not (staged and resolved.parent.parent.name == slug):
            raise ValueError(
                f'cannot confirm snapshot {resolved} as checkpoint '
                f'{expected_checkpoint!r}: no ds41rt manifest and no exact '
                f'{slug}/snapshots/<revision> staging path')
        record['confirmed_via'] = 'hf-cache-dir-name'
    return record


def oracle_gate_passed(checks):
    """Timing may only start when every correctness check passed."""
    return bool(checks) and all(check['passed'] for check in checks)


def invariants_gate_passed(invariants):
    """Replay must be allocation-free, deterministic, and address-stable."""
    return bool(invariants) and all(
        entry['replay_allocations'] == 0
        and entry['replay_deterministic']
        and entry['workspace_addresses_stable']
        for entry in invariants)


def unique_storage_bytes(tensors):
    """Bytes of distinct device storage, merging overlapping/aliased ranges.

    The preplanned buffers deliberately share storage: `_make_mixed_trellis_buffers`
    assigns the SAME tensor to `fc2` and `rotation_gate` (FC2 writes into the
    buffer FC1's rotation output vacates after the activation barrier).  Summing
    per-field nbytes would double-count that buffer, so the workspace is priced
    by merging the actual [data_ptr, data_ptr+nbytes) ranges instead.
    """
    ranges = []
    for tensor in tensors:
        span = int(tensor.numel()) * int(tensor.element_size())
        if span:
            ranges.append((int(tensor.data_ptr()), int(tensor.data_ptr()) + span))
    total, cursor = 0, None
    for begin, end in sorted(ranges):
        if cursor is None or begin > cursor:
            total += end - begin
            cursor = end
        elif end > cursor:
            total += end - cursor
            cursor = end
    return total


def nvidia_smi_selector(value):
    """Normalize a torch device UUID to a valid nvidia-smi ``--id`` selector.

    torch's ``props.uuid`` renders as a bare UUID (no ``GPU-`` prefix) while
    nvidia-smi wants ``GPU-<uuid>``; numeric indices and already-prefixed
    (GPU-/MIG-/UUID-) strings pass through untouched.  An empty value has no
    selector at all.
    """
    text = str(value).strip() if value is not None else ''
    if not text or text == 'None':
        return None
    if text.isdigit():
        return text
    if text.upper().startswith(('GPU-', 'MIG-', 'UUID-')):
        return text
    if re.fullmatch(r'[0-9a-fA-F][0-9a-fA-F-]{7,}', text):
        return 'GPU-' + text
    return text


def numeric_similarity(candidate, reference):
    """Reported-always, gated-never (max-relative stays the only gate):
    relative-L2 error and cosine similarity between two output tensors."""
    c, r = candidate.float(), reference.float()
    return dict(
        relative_l2_error=float((c - r).norm() / r.norm()),
        cosine_similarity=float((c * r).sum() / (c.norm() * r.norm())))


def gpu_identity_snapshot(uuid):
    """Raw per-GPU mode sample; purely audit data, never a timing gate here."""
    selector = nvidia_smi_selector(uuid)
    if not selector:
        return None
    try:
        completed = subprocess.run(
            ['nvidia-smi', '--id', str(selector),
             '--query-gpu=uuid,pstate,power.limit,power.draw,clocks.sm,clocks.mem,'
             'clocks_event_reasons.active', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def load_weights(args, torch, safe_open, ProjectionTrellisTierWeights, prepare_projection_native_trellis_weights, replace, tiers=None, family=None):
    experts, hidden, width, start = 384, 5120, args.intermediate, args.slice_start
    layer_prefix = f'layers.{args.layer}'
    index = json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    tensors = {}; bitmaps = {}
    for expert in range(experts):
        for projection in ['w1', 'w3', 'w2']:
            prefix = f'{layer_prefix}.ffn.experts.{expert}.{projection}'
            for suffix in ['trellis','suh','svh','mcg']:
                name = prefix+'.'+suffix
                with safe_open(args.snapshot/index[name], framework='pt', device='cpu') as f:
                    value = f.get_tensor(name)
                    if suffix == 'mcg':
                        if int(value.item()) & 0xffffffff != MCG_MAGIC:
                            raise ValueError(
                                f'{name}: not an MCG-codebook EXL3 tensor '
                                f'(magic {int(value.item()) & 0xffffffff:#x} != {MCG_MAGIC:#x})')
                        continue
                    if suffix == 'trellis':
                        bitmaps[expert, projection] = value.shape[-1] // 16
                        value = value[start//16:(start+width)//16] if projection == 'w2' else value[:, start//16:(start+width)//16]
                    elif (projection == 'w2' and suffix == 'suh') or (projection != 'w2' and suffix == 'svh'):
                        value = value[start:start+width]
                    tensors[expert, projection, suffix] = value.contiguous().to('cuda')
    # The tier family comes from the checkpoint's own per-tensor widths (and an
    # optional explicit declaration), never from the repository name, revision,
    # or a nominal bits-per-weight reading.  The sampled layer must form the same
    # family the loader builds for the WHOLE checkpoint (`checkpoint_global_family`
    # over the publication manifest), so one layer can never legitimise a family
    # the rest of the checkpoint would not compile.
    global_family, global_count, global_uniform, global_hist = \
        checkpoint_global_family(Path(args.snapshot))
    resolved_tiers, record = resolve_tier_family(
        tiers, bitmaps.values(), global_family=global_family,
        global_audit=dict(projection_count=global_count, uniform=global_uniform,
                          histogram=global_hist))
    if family is not None:
        family.update(record)
    def rotations(proj, suffix):
        return torch.stack([tensors[e,proj,suffix] for e in range(experts)])
    tiers_by_bits=[]
    members_by_tier={}
    for bits in resolved_tiers:
        members = {p: tuple(e for e in range(experts) if bitmaps[e,p] == bits) for p in ['w1','w3','w2']}
        members_by_tier[bits]=members
        def packed(proj):
            shape = (width//16, hidden//16, 16*bits) if proj=='w2' else (hidden//16, width//16, 16*bits)
            return torch.stack([tensors[e,proj,'trellis'] for e in members[proj]]) if members[proj] else torch.empty((0,*shape),dtype=torch.int16,device='cuda')
        tiers_by_bits.append(ProjectionTrellisTierWeights(bits, torch.cat((packed('w1'),packed('w3'))),
            packed('w2'), members['w1'],members['w3'],members['w2']))
    if family is not None:
        family['membership'] = {str(bits): {proj: list(members) for proj, members in
                                            members_by_tier[bits].items()}
                                for bits in resolved_tiers}
    for projection in ['w1', 'w3', 'w2']:
        covered = sorted(e for bits in resolved_tiers for e in members_by_tier[bits][projection])
        if covered != list(range(experts)):
            raise ValueError(
                f'tier membership does not partition all {experts} experts for {projection}')
    prepared = prepare_projection_native_trellis_weights(tuple(tiers_by_bits),
        gate_suh=rotations('w1','suh'), up_suh=rotations('w3','suh'),
        intermediate_rotations=torch.cat((rotations('w1','svh'),rotations('w3','svh'),rotations('w2','suh')),dim=1),
        down_svh=rotations('w2','svh'), activation='silu',params_dtype=torch.bfloat16,
        num_experts=experts, hidden_size=hidden, intermediate_size=width)
    # Empty projection membership is legal. Keep one unreachable physical
    # plane for the native binding's non-null storage ABI; descriptor membership
    # and gate/up counts remain unchanged, so this adds no routed expert.
    padded=[]
    for bits,tier in zip(resolved_tiers,prepared.tiers):
        stride=(width//16)*(hidden//16)*(8*bits)
        updates={}
        if tier.w13.numel()==0:
            updates['w13']=torch.zeros(stride,dtype=torch.int32,device='cuda')
        if tier.w2.numel()==0:
            updates.update(w2=torch.zeros(stride,dtype=torch.int32,device='cuda'),
                w2_global_scale=torch.ones(1,dtype=torch.float32,device='cuda'))
        padded.append(replace(tier,**updates))
    prepared=replace(prepared,tiers=tuple(padded))
    return prepared


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--intermediate', type=int, choices=(512, 640, 768), required=True,
                        help='512/640: legacy TP4 paired widths; 768: disjoint TP3 rank slice')
    parser.add_argument('--slice-start', type=int, default=0,
                        help='TP3 (width 768) admits 0/768/1536 only; legacy widths keep the '
                             'H128-block rule')
    parser.add_argument('--layer', type=int, default=30)
    parser.add_argument('--capacity', type=int, choices=(16, 80), default=16)
    parser.add_argument('--tiers', default=None,
                        help='explicit two-tier family "a,b" (for example 3,4); default '
                             'derives the family from the checkpoint trellis widths and '
                             'fails closed on anything else (never name/bpw inference)')
    parser.add_argument('--checkpoint', default=None,
                        help='require this exact org/repo id; for example '
                             'wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1. Unconfirmable '
                             'snapshots fail closed.')
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


def variant_tile(record, name, fallback):
    """Reported tile for `name`: its own record when present (compiled, legal
    but untimed, or rejected-numerics), else the requested tile."""
    for variant in record['variants']:
        if variant.get('name') == name and 'tile' in variant:
            return variant['tile']
    return fallback


def main(argv=None):
    args = parse_args(argv)
    try:
        layout = resolve_layout(args.intermediate, args.slice_start)
    except ValueError as exc:
        raise SystemExit(f'--slice-start/--intermediate rejected: {exc}') from exc
    declared_tiers = None
    try:
        declared_tiers = parse_tiers(args.tiers)
    except ValueError as exc:
        raise SystemExit(f'--tiers rejected: {exc}') from exc
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            'bench_v41_exl3_tiles is a GPU-only offline experiment (SM121); '
            'no CUDA device is available, so nothing ran')
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 1):
        raise RuntimeError(
            'bench_v41_exl3_tiles targets the Spark SM121 package path, got '
            f'{props.name} (SM{props.major}{props.minor})')
    from safetensors import safe_open
    from b12x.moe.fused_moe.trellis import (
        ProjectionTrellisTierWeights, prepare_projection_native_trellis_weights,
    )
    from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
    from b12x.moe.fused_moe._impl import (
        _projection_mixed_direct_topk_routes, _projection_mixed_tile_config,
    )
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        compile_mixed_trellis, make_mixed_trellis_buffers, bind_mixed_trellis, run_bound_mixed_trellis,
    )
    from dataclasses import fields
    checkpoint = checkpoint_identity(args.snapshot, args.checkpoint)
    tier_family = {}
    prepared = load_weights(args, torch, safe_open, ProjectionTrellisTierWeights,
                            prepare_projection_native_trellis_weights, replace,
                            tiers=declared_tiers, family=tier_family)
    # Route geometry follows the production decision exactly (export_b12x_v41_exl3_aot):
    # the pinned planner's own `_projection_mixed_direct_topk_routes` (two-tier EXL3
    # only; capped by _MIXED_TRELLIS_DIRECT_ROUTE_LIMIT, currently 0 => packed), and
    # the same `ceil(slots / block)` max_m_blocks instead of the earlier floor.
    two_tier = len(tier_family['tiers']) == 2
    direct = bool(_projection_mixed_direct_topk_routes(
        args.capacity, 6, direct_exl3=two_tier))
    if direct:
        route_slots = args.capacity * 6
        max_m_blocks = route_slots
    else:
        route_slots = route_pack_capacity(args.capacity * 6, MOE_BLOCK_SIZE, 384,
                                          topk=6)[1]
        max_m_blocks = (route_slots + MOE_BLOCK_SIZE - 1) // MOE_BLOCK_SIZE
    baseline_tile = _projection_mixed_tile_config(None, hidden_size=5120,
        intermediate_size=args.intermediate, token_count=args.capacity,
        direct_topk_routes=direct)
    specs = plan_variants(args.intermediate, baseline_tile)
    plans = []
    record = dict(scope=__doc__, tool='bench_v41_exl3_tiles',
                  command=[*(argv if argv is not None else sys.argv)],
                  non_policy=True, promotes_default=False,
                  source_revision=_pinned_sparkinfer.REVISION,
                  source=dict(sparkinfer_revision=_pinned_sparkinfer.REVISION,
                              sparkinfer_version=_pinned_sparkinfer.VERSION,
                              source_tree_sha256=_pinned_sparkinfer.LOCK_DATA.get(
                                  'source_tree_sha256')),
                  checkpoint=checkpoint,
                  tier_family=tier_family,
                  gpu=props.name, sm_count=props.multi_processor_count,
                  device=dict(name=props.name, uuid=str(getattr(props, 'uuid', '')),
                              nvidia_smi_selector=nvidia_smi_selector(
                                  getattr(props, 'uuid', '')),
                              compute=[props.major, props.minor],
                              sm_count=props.multi_processor_count,
                              shared_memory_per_block_optin=int(props.shared_memory_per_block_optin),
                              total_memory_bytes=int(getattr(props, 'total_memory', 0))),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  layout=dict(**layout, capacity=args.capacity, experts=384, hidden=5120,
                              top_k=6, routing='direct' if direct else 'packed',
                              direct_topk_routes=direct,
                              route_slots=route_slots, max_m_blocks=max_m_blocks,
                              planner_tile=list(baseline_tile)),
                  configuration=vars(args).copy(),
                  notes=[
                      'offline experiment: writes no config, default, embedded profile, '
                      'or policy resolution',
                      'oracle = production-planner baseline cross-variant gate, eager/graph '
                      'bitwise equality, replay determinism, live-count schedule '
                      'invariance, and quantization-semantics contracts (MCG magic, tier '
                      'membership partition, fail-closed family rule); the pinned source '
                      'ships no independent MCG dequant reference to price against',
                      'blocks_per_sm is recorded as the compiler resolved it; forcing '
                      'residency exists only on the legacy paired TP4 widths and cannot be '
                      'requested on the disjoint TP3 path',
                      'graph timings cover routing, mixed core and reduction for one rank '
                      'slice; TP3 serving sums three rank partials, transport excluded',
                  ],
                  variants=[], results=[], passed=False)
    record['configuration'] = {k: str(v) if isinstance(v, Path) else v for k, v in record['configuration'].items()}

    def save():
        args.output.write_text(json.dumps(record, indent=2) + '\n')

    save()
    for spec in specs:
        if spec.get('skip'):
            record['variants'].append(dict(name=spec['name'], kind=spec['kind'],
                                           tile_requested=list(spec['tile']),
                                           status=spec['skip']))
            print('variant skipped', record['variants'][-1], flush=True)
            save()
            continue
        started = time.monotonic()
        try:
            launch = compile_mixed_trellis(
                    size_m=args.capacity, hidden_size=5120, intermediate_size=args.intermediate,
                    tier0_num_experts=384, tier1_num_experts=384, route_num_experts=384,
                    top_k=6, max_m_blocks=max_m_blocks,
                    sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
                    force_tile_config=spec['tile'],
                    tier0_bits=tier_family['tiers'][0], tier1_bits=tier_family['tiers'][1],
                    route_ids_dtype=torch.int32,
                    **COMPILE_PARITY_KWARGS,
                    direct_topk_routes=direct, force_blocks_per_sm=spec['blocks_per_sm_forced'])
        except ValueError as exc:
            if spec['kind'] != 'controlled-tile' or not is_tile_legality_error(exc):
                raise
            record['variants'].append(dict(name=spec['name'], kind=spec['kind'],
                                           tile_requested=list(spec['tile']),
                                           status='illegal-for-geometry', error=str(exc)))
            print('controlled candidate declined by the pinned planner',
                  record['variants'][-1], flush=True)
            save()
            continue
        buffers = make_mixed_trellis_buffers(launch, device=torch.device('cuda', 0), sms=props.multi_processor_count)
        binding = bind_mixed_trellis(*prepared.tiers, prepared.global_to_combined,
            prepared.descriptor_map, prepared.rotations, launch,
            gate_experts=prepared.gate_counts, up_experts=prepared.up_counts)
        plans.append((spec['name'], spec['kind'], binding, buffers))
        # What the compiler actually resolved: the tile and blocks_per_sm here
        # are read back from the launch, never echoed from the request.
        record['variants'].append(dict(
            name=spec['name'], kind=spec['kind'], tile_requested=list(spec['tile']),
            tile=[launch.fc1_tile_k, launch.fc1_tile_n, launch.fc2_tile_k, launch.fc2_tile_n],
            blocks_per_sm=int(launch.blocks_per_sm),
            blocks_per_sm_forced=spec['blocks_per_sm_forced'],
            tier_bits=[launch.tier0_bits, launch.tier1_bits],
            shared_memory_bytes=int(launch.shared_memory_bytes),
            sms=int(launch.sms),
            workspace_bytes=unique_storage_bytes(
                [getattr(buffers, field.name) for field in fields(buffers)]),
            compile_seconds=time.monotonic() - started, status='compiled'))
        print('compiled', record['variants'][-1], flush=True)
        save()
    if not plans:
        record['failure'] = 'no variant compiled'
        save()
        raise SystemExit('no variant compiled')
    if layout['mode'] == 'tp3-disjoint-width768' and len(plans) < 2:
        record['failure'] = ('the disjoint TP3 race needs the production baseline plus at '
                             'least one legal controlled tile candidate')
        save()
        raise SystemExit(record['failure'])
    residency_pair = [index for index, plan in enumerate(plans)
                      if plan[1] == 'paired-residency']
    workspace_pointers = {name: tuple(int(tensor.data_ptr()) for field in fields(buffers)
                                      for tensor in (getattr(buffers, field.name),))
                          for name, _, _, buffers in plans}
    cases = [(1, 6), (8, 16), (8, 30), (16, 30)]
    if args.capacity == 80:
        cases += [(24, 30), (64, 30)]
    # Formal policy: correctness precedes performance GLOBALLY.  Phase A runs
    # every case's capture/invariants/numerics battery and opens NO timing
    # windows; any structural or baseline failure withholds the whole timing
    # phase.  A CONTROLLED candidate's own numerical failure does not poison
    # the lane — it is classified ('rejected-numerics') and excluded from
    # every timing case (all-or-nothing, never only the failing cases) — the
    # gate itself stays max-relative <= .004, never loosened, with relative-L2
    # and cosine reported alongside.
    numerics_rejected: dict = {}
    global_failures: list = []
    timing_queue: list = []
    for rows, unique in cases:
        torch.manual_seed(410917 + rows + unique)
        x = torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16)
        members = torch.randperm(384, device='cuda')[:unique]
        ids = members[torch.arange(rows * 6, device='cuda').reshape(rows, 6) % unique].to(torch.int32)
        weights = torch.softmax(torch.randn(rows, 6, device='cuda'), dim=1)
        graphs, outputs = [], []
        for name, kind, binding, buffers in plans:
            for _ in range(3):
                run_bound_mixed_trellis(x, weights, ids, binding, buffers)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run_bound_mixed_trellis(x, weights, ids, binding, buffers)
            graphs.append(graph)
            outputs.append(out)
        # Serving invariants before timing: graph replay must not touch the
        # caching allocator, must be bitwise deterministic, and must keep the
        # plan-time (preplanned, capacity-sized) workspace addresses stable.
        invariants = []
        for (name, kind, binding, buffers), graph, output in zip(plans, graphs, outputs):
            graph.replay()
            torch.cuda.synchronize()
            first = output.clone()
            begin_allocations = torch.cuda.memory_stats(0)['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            replay_allocations = int(
                torch.cuda.memory_stats(0)['allocation.all.allocated'] - begin_allocations)
            pointers = tuple(int(tensor.data_ptr()) for field in fields(buffers)
                             for tensor in (getattr(buffers, field.name),))
            invariants.append(dict(
                variant=name, replay_allocations=replay_allocations,
                replay_deterministic=bool(torch.equal(first, output)),
                workspace_addresses_stable=bool(pointers == workspace_pointers[name])))
        case_passed = invariants_gate_passed(invariants)
        checks = []
        anchor_eager = {}
        anchor_inputs = None
        for mutation in range(2):
            if mutation == 0:
                # Snapshot the unmutated first row so the 1-row schedule check
                # compares the same route that produced the full-shape row 0.
                anchor_inputs = (x[:1].clone(), weights[:1].clone(), ids[:1].clone())
            if mutation:
                x.mul_(-.75)
                ids.copy_(ids.roll(1, dims=0).roll(1, dims=1))
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            reference = outputs[0].clone()
            assert torch.isfinite(reference).all() and reference.abs().max() > 0
            if len(residency_pair) == 2:
                # A residency-only change must preserve the whole-tile arithmetic.
                assert torch.equal(outputs[residency_pair[0]], outputs[residency_pair[1]]), \
                    'changing residency changed same-tile output'
            for (name, kind, binding, buffers), output in zip(plans, outputs):
                assert torch.isfinite(output).all() and output.abs().max() > 0
                error = float((output.float() - reference.float()).abs().max() / reference.float().abs().max())
                copied = output.clone()
                eager = run_bound_mixed_trellis(x, weights, ids, binding, buffers)
                torch.cuda.synchronize()
                assert torch.equal(copied, eager), (name, 'graph differs from eager')
                check = dict(variant=name, mutation=mutation, relative_max_error=error,
                             passed=error <= .004,
                             **numeric_similarity(copied, reference))
                if error > .004:
                    indices = (copied.float() - reference.float()).abs().flatten().topk(8).indices
                    check['largest_differences'] = dict(indices=indices.tolist(),
                        reference=reference.flatten()[indices].tolist(), candidate=copied.flatten()[indices].tolist())
                checks.append(check)
                if mutation == 0:
                    anchor_eager[name] = copied[:1].clone()
        # Schedule independence across live row counts under one frozen
        # resolution: the same first route row run as a 1-row batch must agree
        # with its row inside the full live-shape result (the packed route
        # blocks and grid waves differ; the arithmetic may not).
        row_consistency = []
        if rows > 1 and anchor_inputs is not None:
            anchor_x, anchor_weights, anchor_ids = anchor_inputs
            for (name, kind, binding, buffers), full_row in zip(plans, anchor_eager.values()):
                one = run_bound_mixed_trellis(anchor_x.clone(), anchor_weights.clone(),
                                              anchor_ids.clone(), binding, buffers)
                torch.cuda.synchronize()
                error = float((one.float() - full_row.float()).abs().max()
                              / full_row.float().abs().max().clamp_min(1e-12))
                row_consistency.append(dict(variant=name, rows_1_vs_full_max_error=error,
                                            passed=error <= .004,
                                            **numeric_similarity(one, full_row)))
        # Classify per-variant numeric failures; a controlled candidate's own
        # miss is a rejection, never a lane-wide timing veto.  Rejection is
        # all-or-nothing: a candidate that fails ANY case is excluded from
        # EVERY timing case.  The gate stays max-relative <= .004.
        failed_variants = ({c['variant'] for c in checks if not c['passed']}
                           | {e['variant'] for e in row_consistency if not e['passed']})
        if not case_passed:
            global_failures.append(f'invariant failure in case {rows}x{unique}')
        candidate_names = {p[0] for p in plans} - {plans[0][0]}
        if candidate_names and candidate_names <= failed_variants:
            # EVERY controlled candidate disagrees with the production
            # baseline: the shared reference (the baseline) is the suspect,
            # not the whole candidate set.  Treat as a global failure.
            global_failures.append(f'baseline numerics failed in case {rows}x{unique}')
        else:
            # A baseline failure is either the all-disagree branch above or a
            # genuine baseline-only miss; both fail closed.
            for name in sorted(failed_variants):
                if name == plans[0][0]:
                    global_failures.append(f'baseline numerics failed in case {rows}x{unique}')
                else:
                    worst = max((c['relative_max_error'] for c in checks
                                 if c['variant'] == name), default=None)
                    numerics_rejected.setdefault(name, []).append(dict(
                        case=[rows, unique], worst_relative_max_error=worst))
        record['results'].append(dict(rows=rows, distinct_experts=unique, checks=checks,
                                      invariants=invariants, row_consistency=row_consistency,
                                      same_tile_residency_bitwise=(
                                          True if len(residency_pair) == 2 else None),
                                      timings=None))
        timing_queue.append(dict(rows=rows, unique=unique, graphs=graphs, outputs=outputs,
                                 result=record['results'][-1]))
        print('case correctness done', rows, unique, 'global_failures', len(global_failures),
              'rejected', sorted(numerics_rejected), flush=True)
        save()
    if numerics_rejected:
        for variant in record['variants']:
            if variant.get('name') in numerics_rejected and variant.get('status') == 'compiled':
                variant['status'] = 'rejected-numerics'
                variant['numerics_rejections'] = numerics_rejected[variant['name']]
    record['numerics_rejected'] = {name: reasons for name, reasons in sorted(numerics_rejected.items())}
    timed_plans = [p for p in plans if p[0] not in numerics_rejected]
    if global_failures:
        record['failure'] = ('global correctness failure; timing withheld for every case: '
                             + '; '.join(global_failures))
        save()
        raise SystemExit(record['failure'])
    if not timed_plans:
        record['failure'] = 'no timing-eligible variant survived the numerical gate'
        save()
        raise SystemExit(record['failure'])
    # Phase B: the whole battery passed, now open the timing windows only.
    # Rejection is all-or-nothing per candidate: one numerical failure anywhere
    # means the candidate is never timed in any case, so `timed_plans` (and the
    # samples/orders sized beside it) must be keyed off `numerics_rejected`, not
    # off which individual cases happened to fail.
    for context in timing_queue:
        graphs, outputs, result = context['graphs'], context['outputs'], context['result']
        rows, unique = context['rows'], context['unique']
        # Balanced forward/reverse order; raw samples are retained. Stable graphs
        # replay repeatedly without resolving kernels or allocating workspaces.
        timed_graphs = [context['graphs'][i] for i, p in enumerate(plans)
                        if p[0] not in numerics_rejected]
        samples = [[] for _ in timed_plans]
        orders = [list(range(len(timed_plans))), list(range(len(timed_plans)-1, -1, -1))] * 3
        # Bracket the actual timing window: clocks_before is sampled immediately
        # before any replay and clocks_after immediately after the last sample, so
        # the pair describes the machine state the numbers were taken in.
        clocks_before = gpu_identity_snapshot(record['device']['uuid'])
        timing_begin = torch.cuda.memory_stats(0)['allocation.all.allocated']
        for order in orders:
            for index in order:
                for _ in range(5):
                    timed_graphs[index].replay()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(100):
                    timed_graphs[index].replay()
                end.record()
                end.synchronize()
                samples[index].append(begin.elapsed_time(end) * 10)
        timing_allocations = int(
            torch.cuda.memory_stats(0)['allocation.all.allocated'] - timing_begin)
        clocks_after = gpu_identity_snapshot(record['device']['uuid'])
        result.update(clocks_before=clocks_before,
                      timing_order=[[timed_plans[index][0] for index in order] for order in orders],
                      timing_iterations=100,
                      timing_allocations=timing_allocations,
                      clocks_after=clocks_after,
                      timings=[dict(variant=p[0], tile=variant_tile(record, p[0], p[1]),
                                    samples_us=s, median_us=statistics.median(s),
                                    min_us=min(s), max_us=max(s))
                               for p, s in zip(timed_plans, samples)])
        print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'invariants')}), flush=True)
        save()
        del graphs, outputs
    record['passed'] = (not global_failures
                        and all(check['passed'] for result in record['results']
                                for check in result['checks']
                                if check['variant'] not in numerics_rejected)
                        and all(entry['passed'] for result in record['results']
                                for entry in result.get('row_consistency', [])
                                if entry['variant'] not in numerics_rejected)
                        and all(result['invariants'] and invariants_gate_passed(result['invariants'])
                                for result in record['results'])
                        and all(result['timings'] is not None for result in record['results']))
    save()


if __name__ == '__main__':
    main()
