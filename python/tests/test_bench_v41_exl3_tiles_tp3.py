"""CPU contract tests for the disjoint TP3 mode of the EXL3 tile harness.

The harness is GPU-only; this file owns everything that must be true *before*
a Spark run: the exact TP3 rank-slice contract (starts 0/768/1536, whole H128
blocks, no padding), the fail-closed tier-family rule (declared or derived
from real trellis widths, never from the repository name or nominal bpw), the
variant race (production planner tile plus controlled candidates, paired
residency never smuggled into TP3), checkpoint identity binding, the gates
that block timing, and the unchanged legacy TP4 race. The module is loaded
with a fake `_pinned_sparkinfer` so no source pin or CUDA is required.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / 'python' / 'tools'
TARGET = 'wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1'

def load_harness():
    pinned = types.ModuleType('_pinned_sparkinfer')
    pinned.REVISION = 'pinned-for-tests'
    pinned.VERSION = '0.0.0'
    pinned.LOCK_DATA = {'source_tree_sha256': 'tree', 'revision': 'pinned-for-tests'}
    spec = importlib.util.spec_from_file_location(
        'bench_v41_exl3_tiles', TOOLS / 'bench_v41_exl3_tiles.py')
    with patch.dict(sys.modules, {'_pinned_sparkinfer': pinned}):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module

harness = load_harness()

class Tp3SliceContractTests(unittest.TestCase):
    def test_exact_rank_starts_are_disjoint_unpadded_six_block_thirds(self):
        for start, rank in ((0, 0), (768, 1), (1536, 2)):
            with self.subTest(start=start):
                layout = harness.resolve_layout(768, start)
                self.assertEqual(layout['mode'], 'tp3-disjoint-width768')
                self.assertEqual(layout['rank'], rank)
                self.assertEqual(layout['whole_h128_blocks'], 6)
                self.assertFalse(layout['padded'])
        starts = [harness.resolve_layout(768, s)['slice_start']
                  for s in harness.TP3_RANK_STARTS]
        covered = sum(768 for _ in starts)
        self.assertEqual(covered, harness.TOTAL_INTERMEDIATE)
        self.assertEqual(starts, [0, 768, 1536])

    def test_non_rank_starts_fail_closed(self):
        # 640/1280 look legal under the legacy %128 rule; they are padding or
        # overlap masquerading as a TP3 rank and must not race.
        for start in (128, 640, 1280, 1664, 1792, 2304, -768):
            with self.subTest(start=start):
                with self.assertRaises(ValueError):
                    harness.resolve_layout(768, start)

    def test_legacy_tp4_rules_are_unchanged(self):
        self.assertEqual(harness.resolve_layout(640, 1280)['mode'], 'tp4-paired-legacy')
        self.assertEqual(harness.resolve_layout(512, 1792)['mode'], 'tp4-paired-legacy')
        self.assertEqual(harness.resolve_layout(640, 1664)['mode'], 'tp4-paired-legacy')
        with self.assertRaises(ValueError):
            harness.resolve_layout(512, 1920)   # past 2304
        with self.assertRaises(ValueError):
            harness.resolve_layout(640, 100)    # off the H128 grid
        with self.assertRaises(ValueError):
            harness.resolve_layout(1024, 0)     # unsupported width
        with self.assertRaises(ValueError):
            harness.resolve_layout(1152, 0)

class TierFamilyTests(unittest.TestCase):
    def test_family_is_derived_from_real_widths_not_names(self):
        tiers, record = harness.resolve_tier_family(None, [3, 4, 3])
        self.assertEqual(tiers, (3, 4))
        self.assertEqual(record['resolution'], 'derived-from-checkpoint-trellis-widths')
        self.assertEqual(record['observed_widths'], [3, 4])
        # A uniform-K4 checkpoint belongs to [4, 5]; the K3.25 *name* must never
        # talk it into [3, 4].
        self.assertEqual(harness.resolve_tier_family(None, [4])[0], (4, 5))
        self.assertEqual(harness.resolve_tier_family(None, [3])[0], (3, 4))
        self.assertEqual(harness.resolve_tier_family(None, [5])[0], (4, 5))
        self.assertEqual(harness.resolve_tier_family(None, [2])[0], (2, 3))

    def test_declared_tiers_must_equal_the_checkpoint_family(self):
        tiers, record = harness.resolve_tier_family((3, 4), [3, 4])
        self.assertEqual(tiers, (3, 4))
        self.assertEqual(record['resolution'],
                         'declared-and-confirmed-by-checkpoint-trellis-widths')
        with self.assertRaises(ValueError):
            harness.resolve_tier_family((3, 4), [4])   # uniform K4 is family [4,5]
        with self.assertRaises(ValueError):
            harness.resolve_tier_family((4, 5), [3, 4])

    def test_families_this_two_tier_path_cannot_run_fail_closed(self):
        with self.assertRaises(ValueError):
            harness.resolve_tier_family(None, [2, 3, 4])
        with self.assertRaises(ValueError):
            harness.resolve_tier_family(None, [3, 6])
        with self.assertRaises(ValueError):
            harness.resolve_tier_family(None, [])

    def test_parse_tiers_cli_surface(self):
        self.assertIsNone(harness.parse_tiers(None))
        self.assertIsNone(harness.parse_tiers('auto'))
        self.assertEqual(harness.parse_tiers('3,4'), (3, 4))
        for bad in ('3', '4,3', '3,3', 'a,b', '3,4,5', '3'):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    harness.parse_tiers(bad)

class VariantRaceTests(unittest.TestCase):
    def test_tp3_m16_races_planner_tile_against_both_k64_widths(self):
        # _projection_mixed_tile_config(None, token_count=16) resolves
        # (128,128,128,128); the k128 candidate is then the production
        # geometry itself and must not be double-raced.
        specs = harness.plan_variants(768, (128, 128, 128, 128))
        raced = {s['name']: s for s in specs if 'skip' not in s}
        skipped = {s['name'] for s in specs if 'skip' in s}
        self.assertEqual(raced['baseline']['tile'], (128, 128, 128, 128))
        self.assertEqual(raced['baseline']['kind'], 'production-planner')
        self.assertEqual(set(raced) - {'baseline'}, {'k64-n256', 'k64-n128'})
        self.assertEqual(skipped, {'k128-n128'})
        self.assertEqual(raced['k64-n256']['tile'], (64, 256, 64, 256))
        self.assertEqual(raced['k64-n128']['tile'], (64, 128, 64, 128))

    def test_tp3_m80_races_planner_tile_against_k64n128_and_the_m16_geometry(self):
        specs = harness.plan_variants(768, (64, 256, 64, 256))
        raced = {s['name']: s for s in specs if 'skip' not in s}
        skipped = {s['name'] for s in specs if 'skip' in s}
        self.assertEqual(raced['baseline']['tile'], (64, 256, 64, 256))
        self.assertEqual(skipped, {'k64-n256'})
        self.assertEqual(set(raced) - {'baseline'}, {'k64-n128', 'k128-n128'})

    def test_tp3_never_forces_blocks_per_sm(self):
        """Paired-only residency must not leak into a disjoint TP3 race."""
        for tile in ((64, 256, 64, 256), (128, 128, 128, 128)):
            for spec in harness.plan_variants(768, tile):
                self.assertIsNone(spec['blocks_per_sm_forced'], spec['name'])
                self.assertNotEqual(spec['kind'], 'paired-residency')

    def test_legacy_race_is_byte_for_byte_the_historical_one(self):
        specs = harness.plan_variants(640, (64, 128, 64, 128))
        self.assertEqual([(s['name'], s['tile'], s['blocks_per_sm_forced'])
                          for s in specs],
                         [('baseline', (64, 128, 64, 128), 1),
                          ('k64-one-block', (64, 128, 64, 128), 1),
                          ('k64-two-blocks', (64, 128, 64, 128), 2)])
        self.assertEqual([s['name'] for s in harness.plan_variants(512, (64, 256, 64, 256))],
                         ['baseline', 'k64-one-block', 'k64-two-blocks'])

    def test_controlled_candidates_declined_by_the_compiler_are_narrowly_skipped(self):
        self.assertTrue(harness.is_tile_legality_error(
            'size_n must be divisible by tile_n'))
        self.assertTrue(harness.is_tile_legality_error(
            'force_tile_config FC1/FC2 thread counts must match, got (64, 256) vs 128'))
        self.assertTrue(harness.is_tile_legality_error(
            'mixed Trellis shared-memory requirement exceeds the device limit'))
        # Anything else is a real failure and must abort, not be recorded
        # as "illegal".
        self.assertFalse(harness.is_tile_legality_error('CUDA error: illegal memory access'))
        self.assertFalse(harness.is_tile_legality_error('descriptor rows must be 4'))

class CheckpointIdentityTests(unittest.TestCase):
    """The staged-manifest check must have sync-helper semantics: exact schema,
    exact model_id, the manifest revision (or `commit` field) equal to the
    snapshots/<revision> directory, and — for the content-addressed scheme —
    the canonical sha256({"schema","files"}) re-deriving that revision. A
    git-commit revision (40-hex, e.g. the real K3.25 staging) is legitimately
    not that manifest hash, so identity there rests on the exact field
    equality instead. All deviations fail closed."""

    SCHEMA = 'ds41rt-hf-staged-snapshot-v1'

    def _snapshot(self, root: Path, files: dict[str, str] | None = None):
        root.mkdir(parents=True, exist_ok=True)
        (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {}}))
        for name, text in (files or {}).items():
            (root / name).write_text(text)
        return root

    def _staged(self, temp, revision, *, model_id=TARGET, schema=None, files=None,
                revision_field='revision', manifest_revision=None, write_manifest=True):
        model_root = Path(temp) / 'models--wrldsuksgo2mars--DeepSeek-V4.1-EXL3-K3.25-v1'
        snap = model_root / 'snapshots' / revision
        self._snapshot(snap)
        if write_manifest:
            manifests = model_root / 'ds41rt-manifests'
            manifests.mkdir(exist_ok=True)
            payload = {'schema': self.SCHEMA if schema is None else schema,
                       'model_id': model_id, 'files': [
                           {'name': 'model.safetensors.index.json', 'sha256': 'x'}]
                       if files is None else files}
            payload[revision_field] = revision if manifest_revision is None \
                else manifest_revision
            (manifests / f'{revision}.json').write_text(json.dumps(payload))
        return snap

    @staticmethod
    def _content_revision(files):
        return hashlib.sha256(json.dumps(
            {'schema': CheckpointIdentityTests.SCHEMA, 'files': files},
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def test_records_index_digest_without_any_expected_id(self):
        with self.subTest('tmp'):
            with tempfile.TemporaryDirectory() as temp:
                snap = self._snapshot(Path(temp))
                record = harness.checkpoint_identity(snap)
                self.assertIsNone(record['requested'])
                self.assertIsNone(record['confirmed_via'])
                self.assertEqual(
                    record['index_sha256'],
                    hashlib.sha256((snap / 'model.safetensors.index.json')
                                   .read_bytes()).hexdigest())

    def test_missing_index_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                harness.checkpoint_identity(Path(temp), TARGET)

    def test_symlinked_snapshot_directory_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            real = self._snapshot(Path(temp) / 'real')
            link = Path(temp) / 'link'
            os.symlink(real, link)
            with self.assertRaises(ValueError):
                harness.checkpoint_identity(link)

    def test_unconfirmable_snapshot_fails_closed_when_an_id_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            snap = self._snapshot(Path(temp))
            with self.assertRaises(ValueError):
                harness.checkpoint_identity(snap, TARGET)

    def test_hf_cache_slug_confirms_a_git_commit_staged_snapshot(self):
        # The real K3.25 staging form: 40-hex commit directory, no manifest.
        with tempfile.TemporaryDirectory() as temp:
            revision = 'cfd4ca1d1934a8e81dd2d7515598d4ce288e8b88'
            snap = self._staged(temp, revision, write_manifest=False)
            record = harness.checkpoint_identity(snap, TARGET)
            self.assertEqual(record['confirmed_via'], 'hf-cache-dir-name')
            self.assertEqual(record['revision'], revision)
            # A wrong-shaped layout (no `snapshots/` parent) must NOT confirm.
            stray = Path(temp) / 'backup' / 'models--wrldsuksgo2mars--DeepSeek-V4.1-EXL3-K3.25-v1'
            self._snapshot(stray / revision)
            with self.assertRaises(ValueError):
                harness.checkpoint_identity(stray / revision, TARGET)

    def test_verified_manifest_binds_model_and_revision_with_sync_semantics(self):
        with tempfile.TemporaryDirectory() as temp:
            files = [{'name': 'model.safetensors.index.json', 'sha256': 'deadbeef'}]
            revision = self._content_revision(files)
            snap = self._staged(temp, revision, files=files)
            record = harness.checkpoint_identity(snap, TARGET)
            self.assertEqual(record['confirmed_via'], 'ds41rt-manifest-verified')
            self.assertTrue(record['manifest_verified'])
            self.assertEqual(record['manifest_revision_scheme'], 'content-sha256')
            self.assertEqual(record['manifest_content_sha256'], revision)

    def test_commit_field_is_accepted_where_a_publisher_used_that_name(self):
        with tempfile.TemporaryDirectory() as temp:
            revision = '076c0dcf8843f2d1a5ba55d84f369017e77914b9'   # 40-hex scheme
            snap = self._staged(temp, revision, revision_field='commit')
            record = harness.checkpoint_identity(snap, TARGET)
            self.assertEqual(record['manifest_revision_scheme'], 'git-commit')
            self.assertTrue(record['manifest_verified'])

    def test_manifest_deviations_fail_closed(self):
        cases = ['schema', 'revision', 'hash', 'files', 'model']
        with tempfile.TemporaryDirectory() as temp:
            for case in cases:
                with self.subTest(case=case):
                    files = [{'name': 'index', 'sha256': 'a'}]
                    revision = self._content_revision(files)
                    kwargs = {'files': files}
                    if case == 'schema':
                        kwargs['schema'] = 'ds41rt.staged-artifact.v1'
                    elif case == 'revision':
                        # 40-hex (git-commit) dir so the scheme check cannot
                        # mask this one: only the field mismatch may fire.
                        revision = 'd' * 40
                        kwargs['manifest_revision'] = 'e' * 40
                    elif case == 'hash':
                        files = [{'name': 'index', 'sha256': 'b'}]  # hashes elsewhere
                        kwargs['files'] = files
                    elif case == 'files':
                        kwargs['files'] = 'not-a-list'
                    elif case == 'model':
                        kwargs['model_id'] = 'someone/else'
                    snap = self._staged(temp, revision, **kwargs)
                    with self.assertRaises(ValueError):
                        harness.checkpoint_identity(snap, TARGET)

    def test_manifest_without_model_identity_is_fatal_even_without_expected_id(self):
        with tempfile.TemporaryDirectory() as temp:
            files = [{'name': 'index', 'sha256': 'a'}]
            snap = self._staged(temp, self._content_revision(files),
                                files=files, model_id='')
            with self.assertRaises(ValueError):
                harness.checkpoint_identity(snap)

    def test_nominal_quantize_bits_are_recorded_never_used(self):
        with tempfile.TemporaryDirectory() as temp:
            snap = self._snapshot(Path(temp) / 'snap0', )
            (snap / 'quantize_config.json').write_text(json.dumps(
                {'bits': 2, 'checkpoint_format': 'exl3', 'codebook': 'mcg'}))
            record = harness.checkpoint_identity(snap)
            self.assertEqual(record['nominal_quantize_metadata']['bits'], 2)
            # The identity path never touches the family rule; widths decide it.
            self.assertEqual(harness.resolve_tier_family(
                None, [3, 4, 3])[0], (3, 4))

    def test_malformed_expected_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            snap = self._snapshot(Path(temp))
            for bad in ('no-slash', 'a/b/c', ''):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    harness.checkpoint_identity(snap, bad)

    def test_refs_main_pointer_must_name_the_benchmarked_revision(self):
        revision = 'cfd4ca1d1934a8e81dd2d7515598d4ce288e8b88'
        with tempfile.TemporaryDirectory() as temp:
            snap = self._staged(temp, revision, write_manifest=False)
            refs = snap.parent.parent / 'refs'
            refs.mkdir(exist_ok=True)
            # Matching pointer: confirms and records the binding.
            (refs / 'main').write_text(revision + '\n')
            record = harness.checkpoint_identity(snap, TARGET)
            self.assertTrue(record['refs_main_matches'])
            # Stale pointer: the cache's current checkout is not this snapshot.
            (refs / 'main').write_text('0' * 40)
            with self.assertRaises(ValueError):
                harness.checkpoint_identity(snap, TARGET)

    def test_manifest_may_certify_no_files(self):
        with tempfile.TemporaryDirectory() as temp:
            snap = self._staged(temp, 'e' * 40, files=[])
            with self.assertRaises(ValueError):
                harness.checkpoint_identity(snap, TARGET)

class CliContractTests(unittest.TestCase):
    BASE = ['--snapshot', '/nonexistent', '--intermediate', '768',
            '--slice-start', '0', '--output', '/tmp/out.json']

    def test_defaults_keep_the_legacy_surface(self):
        args = harness.parse_args(['--snapshot', 's', '--intermediate', '640',
                                   '--output', 'o'])
        self.assertEqual(args.slice_start, 0)
        self.assertEqual(args.layer, 30)
        self.assertEqual(args.capacity, 16)
        self.assertIsNone(args.tiers)
        self.assertIsNone(args.checkpoint)

    def test_admits_intermediate_768_and_still_rejects_other_widths(self):
        self.assertEqual(harness.parse_args(self.BASE).intermediate, 768)
        with self.assertRaises(SystemExit):
            harness.parse_args(['--snapshot', 's', '--intermediate', '1152',
                                '--output', 'o'])
        with self.assertRaises(SystemExit):
            harness.parse_args(['--snapshot', 's', '--intermediate', '384',
                                '--output', 'o'])

    def test_capacity_choices_are_unchanged(self):
        with self.assertRaises(SystemExit):
            harness.parse_args(self.BASE + ['--capacity', '64'])

    def test_bad_slice_and_tiers_exit_before_any_gpu_or_weight_work(self):
        for start in ('128', '640', '1792'):
            argv = ['--snapshot', '/nonexistent', '--intermediate', '768',
                    '--slice-start', start, '--output', '/tmp/out.json']
            with self.subTest(start=start), self.assertRaises(SystemExit) as caught:
                harness.main(argv)
            self.assertIn('slice', str(caught.exception).lower())
        with self.assertRaises(SystemExit) as caught:
            harness.main(self.BASE + ['--tiers', '4,3'])
        self.assertIn('tiers', str(caught.exception))

    def test_off_gpu_fails_cleanly_and_before_weight_reads(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch not importable')
        if torch.cuda.is_available() and \
                (torch.cuda.get_device_properties(0).major,
                 torch.cuda.get_device_properties(0).minor) == (12, 1):
            self.skipTest('host is the SM121 target; contract test would race GPUs')
        with self.assertRaises(RuntimeError) as caught:
            harness.main(self.BASE + ['--checkpoint', TARGET])
        self.assertIn('SM121', str(caught.exception))
        self.assertFalse(Path('/tmp/out.json').exists())

class GateTests(unittest.TestCase):
    def _invariant(self, allocations=0, deterministic=True, stable=True):
        return dict(variant='baseline', replay_allocations=allocations,
                    replay_deterministic=deterministic,
                    workspace_addresses_stable=stable)

    def test_invariants_gate_blocks_replay_allocations(self):
        self.assertTrue(harness.invariants_gate_passed([self._invariant()]))
        self.assertFalse(harness.invariants_gate_passed([]))
        self.assertFalse(harness.invariants_gate_passed([self._invariant(allocations=1)]))
        self.assertFalse(harness.invariants_gate_passed([self._invariant(deterministic=False)]))
        self.assertFalse(harness.invariants_gate_passed([self._invariant(stable=False)]))
        self.assertFalse(harness.invariants_gate_passed(
            [self._invariant(), self._invariant(allocations=3)]))

    def test_oracle_gate_blocks_timing_until_every_check_passes(self):
        self.assertTrue(harness.oracle_gate_passed([{'passed': True}]))
        self.assertFalse(harness.oracle_gate_passed([]))
        self.assertFalse(harness.oracle_gate_passed([{'passed': True}, {'passed': False}]))

class NonPolicyContractTests(unittest.TestCase):
    SOURCE = (TOOLS / 'bench_v41_exl3_tiles.py').read_text()

    def test_top_level_imports_stay_cpu_only(self):
        import ast
        tree = ast.parse(self.SOURCE)
        tops = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                tops.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                tops.add((node.module or '').split('.')[0])
        self.assertLessEqual(tops, {'argparse', 'dataclasses', 'hashlib', 'json',
                                    'pathlib', 're', 'statistics', 'subprocess',
                                    'sys', 'time', '_pinned_sparkinfer'})
        self.assertNotIn('torch', tops)
        self.assertNotIn('b12x', tops)

    def test_no_policy_context_or_profile_writing_apis(self):
        for forbidden in ('PolicyContext', 'get_auto_policy', 'PREPLANNED_ONLY',
                          'HEURISTIC_ONLY', 'generate_gpu_profile', 'plan_tp_moe',
                          'policy_override'):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.SOURCE)

    def test_paired_sibling_call_signature_is_preserved(self):
        import inspect
        params = list(inspect.signature(harness.load_weights).parameters)
        self.assertEqual(params[:6], ['args', 'torch', 'safe_open',
                                      'ProjectionTrellisTierWeights',
                                      'prepare_projection_native_trellis_weights',
                                      'replace'])
        for name in ('tiers', 'family'):
            self.assertIsNone(inspect.signature(harness.load_weights).parameters[name].default)

    def test_record_declares_its_non_policy_stance(self):
        # The run record carries the stance explicitly; grep the builder.
        self.assertIn('non_policy=True', self.SOURCE)
        self.assertIn('promotes_default=False', self.SOURCE)



class SharedFamilyRuleTests(unittest.TestCase):
    """The family rule lives in exactly ONE python place for all tools."""

    def test_harness_and_qualify_oracle_share_one_family_instance(self):
        def load(module_name, filename):
            pinned = types.ModuleType('_pinned_sparkinfer')
            pinned.REVISION = 'pinned-for-tests'
            pinned.VERSION = '0.0.0'
            pinned.LOCK_DATA = {}
            spec = importlib.util.spec_from_file_location(module_name, TOOLS / filename)
            with patch.dict(sys.modules, {'_pinned_sparkinfer': pinned}):
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            return module

        oracle = load('qualify_v41_exl3_aot_contract', 'qualify_v41_exl3_aot.py')
        # `patch.dict` tears the registry entry down when a load context
        # exits, so cross-load identity is proven by the code object's source
        # file (both tools resolve the rule from the one shared file), never by
        # a paraphrase left behind in either tool.
        family_file = str((TOOLS / 'v41_exl3_family.py').resolve())
        for name in ('expected_decoder_family', 'checkpoint_global_family'):
            self.assertEqual(getattr(harness, name).__code__.co_filename,
                             family_file, f'harness.{name}')
            self.assertEqual(getattr(oracle, name).__code__.co_filename,
                             family_file, f'oracle.{name}')
        # Same within one process: the shared loader is a canonical instance.
        shared = oracle._V41_EXL3_FAMILY
        self.assertIs(oracle.checkpoint_global_family, shared.checkpoint_global_family)
        self.assertIs(oracle.expected_decoder_family, shared.expected_decoder_family)
        # No tool file re-defines the extracted rule.
        for filename in ('bench_v41_exl3_tiles.py', 'qualify_v41_exl3_aot.py'):
            self.assertNotIn('def expected_decoder_family',
                             (TOOLS / filename).read_text())

    def test_widths_must_be_plain_ints_and_bool_is_refused(self):
        for bad in ([True], [3, False], [3.0], ['3'], [None]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                harness.expected_decoder_family(bad)

class SampledAgainstGlobalFamilyTests(unittest.TestCase):
    def test_matching_global_is_recorded_both_ways(self):
        tiers, record = harness.resolve_tier_family(
            None, [3, 4], global_family=[3, 4],
            global_audit=dict(projection_count=47232, uniform=False,
                              histogram={'3': 35424, '4': 11808}))
        self.assertEqual(tiers, (3, 4))
        self.assertEqual(record['sampled_layer_family'], [3, 4])
        self.assertEqual(record['checkpoint_global_family'], [3, 4])
        self.assertEqual(record['checkpoint_global']['projection_count'], 47232)
        self.assertEqual(record['resolution'],
                         'derived-from-checkpoint-trellis-widths-matching-global')

    def test_a_sampled_layer_may_not_legitimise_a_different_global_family(self):
        # Uniform-K4 layer inside a globally mixed [3,4] checkpoint: sampled
        # [4,5] vs global [3,4] must fail closed.
        with self.assertRaises(ValueError) as caught:
            harness.resolve_tier_family(None, [4], global_family=[3, 4])
        self.assertIn('checkpoint-global', str(caught.exception))
        with self.assertRaises(ValueError):
            harness.resolve_tier_family((4, 5), [4], global_family=[3, 4])

    def test_declaration_path_names_the_global_confirmation(self):
        _, record = harness.resolve_tier_family((3, 4), [3, 4], global_family=[3, 4])
        self.assertEqual(record['resolution'],
                         'declared-and-confirmed-matching-checkpoint-global')

class UniqueStorageBytesTests(unittest.TestCase):
    class _T:
        def __init__(self, ptr, numel, element_size):
            self._p, self._n, self._e = ptr, numel, element_size

        def data_ptr(self):
            return self._p

        def numel(self):
            return self._n

        def element_size(self):
            return self._e

    def test_aliased_views_price_once(self):
        arena = [self._T(1000, 100, 4),          # 1000..1400 (arena owner)
                 self._T(1040, 10, 4),           # view inside the arena
                 self._T(1400, 50, 4),           # adjacent separate allocation
                 self._T(0, 0, 4)]               # empty placeholder
        self.assertEqual(harness.unique_storage_bytes(arena), 400 + 200)

    def test_identical_fields_count_once(self):
        dup = [self._T(2000, 16, 2), self._T(2000, 16, 2)]
        self.assertEqual(harness.unique_storage_bytes(dup), 32)

    def test_partial_overlap_merges(self):
        t = [self._T(0, 10, 1), self._T(5, 10, 1)]
        self.assertEqual(harness.unique_storage_bytes(t), 15)



class LegalityMarkerSourceTests(unittest.TestCase):
    """Tripwire: every marker substring must still occur in the PINNED
    sparkinfer sources whose rejections the harness classifies as geometry
    declines.  A vanished marker means the pin moved past the audit scope and
    the classification must be re-derived, not silently under-matched."""

    W4A16 = (ROOT / 'third_party' / 'sparkinfer' / 'b12x' / 'moe' / '_shared'
             / 'kernels' / 'w4a16')

    def test_markers_exist_in_pinned_kernel_sources(self):
        if not self.W4A16.is_dir():
            self.skipTest('pinned sparkinfer tree is not checked out')
        blob = ''.join(p.read_text(errors='replace')
                       for p in sorted(self.W4A16.rglob('*.py'))).lower()
        for marker in harness.TILE_LEGALITY_MARKERS:
            with self.subTest(marker=marker):
                self.assertIn(marker.lower(), blob)


class FamilyLoaderFailureTests(unittest.TestCase):
    """A failed import of the shared family module must leave NO half-executed
    registration behind in either tool's loader — while a successful load is
    never popped.  Both loaders share this contract; both are exercised."""

    NAME = 'ds41rt_v41_exl3_family'

    def _exercise(self, tool, tool_filename):
        saved_file = tool.__file__
        with tempfile.TemporaryDirectory() as temp:
            broken = Path(temp) / 'v41_exl3_family.py'
            broken.write_text('raise RuntimeError("family import boom")\n')
            tool.__file__ = str(Path(temp) / tool_filename)
            saved = sys.modules.pop(self.NAME, None)
            try:
                with self.assertRaises(RuntimeError):
                    tool._load_v41_exl3_family()
                self.assertNotIn(self.NAME, sys.modules)   # no partial module
                broken.write_text('marker = "clean-reload"\n')
                loaded = tool._load_v41_exl3_family()      # success registers
                self.assertIs(sys.modules.get(self.NAME), loaded)
            finally:
                if saved is not None:
                    sys.modules[self.NAME] = saved
                    self.assertIs(tool._load_v41_exl3_family(), saved)
                else:
                    sys.modules.pop(self.NAME, None)
                tool.__file__ = saved_file

    def test_harness_loader_leaves_no_partial_module_after_failed_import(self):
        self._exercise(harness, 'bench_v41_exl3_tiles.py')

    def test_qualify_loader_leaves_no_partial_module_after_failed_import(self):
        pinned = types.ModuleType('_pinned_sparkinfer')
        pinned.REVISION = 'p'
        pinned.VERSION = '0'
        pinned.LOCK_DATA = {}
        spec = importlib.util.spec_from_file_location(
            'qualify_loader_failure', TOOLS / 'qualify_v41_exl3_aot.py')
        with patch.dict(sys.modules, {'_pinned_sparkinfer': pinned}):
            oracle = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(oracle)
        self._exercise(oracle, 'qualify_v41_exl3_aot.py')


if __name__ == '__main__':
    unittest.main()
