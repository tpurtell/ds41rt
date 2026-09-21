"""Tier/quant semantics gates for the EXL3 exact-AOT numeric oracle.

The oracle compares a compiled AOT module against B12x on real checkpoint tiles.
These tests cover only its pure contract logic: which serving family a checkpoint's
real trellis widths form, and whether an export may claim it. The family rule mirrors
`decoder_family` in `rust/crates/ds41rt-loader/src/v41_exl3.rs`, so one test reads
that source and fails if the loader's rule moves underneath us.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'exl3_aot_oracle', ROOT / 'python' / 'tools' / 'qualify_v41_exl3_aot.py')
LOADER_SOURCE = ROOT / 'rust' / 'crates' / 'ds41rt-loader' / 'src' / 'v41_exl3.rs'
ORACLE_SOURCE = ROOT / 'python' / 'tools' / 'qualify_v41_exl3_aot.py'

PROJECTIONS = ('w1', 'w3', 'w2')


def load_module():
    pinned = ModuleType('_pinned_sparkinfer')
    pinned.REVISION = 'pinned-for-tests'
    with patch.dict(sys.modules, {'_pinned_sparkinfer': pinned}):
        module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(module)
    return module


oracle = load_module()


def bitmaps(widths):
    """`widths[expert]` is one width for all projections or a per-projection map."""
    result = {}
    for expert, value in enumerate(widths):
        per = dict(zip(PROJECTIONS, [value] * 3)) if isinstance(value, int) else value
        for projection, width in per.items():
            result[(expert, projection)] = width
    return result


class DecoderFamilyTests(unittest.TestCase):
    def test_loader_rule_is_mirrored_not_paraphrased(self):
        """Tripwire: if the Rust family rule moves, this oracle must be re-synced."""
        # Semantic, not formatting: whitespace is stripped from both sides, so
        # rustfmt, a re-wrap or a comment cannot break the tripwire while the rule
        # itself still has to read exactly as the oracle assumes.
        stripped = ''.join(LOADER_SOURCE.read_text().split())
        # Scoped to decoder_family itself: the same `.contains(b)` text appears at
        # two unrelated sites in the loader, so a whole-file grep stays green even
        # if this function's range check is deleted. Anchoring makes it a tripwire.
        start = stripped.index('pub(crate)fndecoder_family(')
        rest = stripped[start + len('pub(crate)fndecoder_family('):]
        end = rest.index('pub(crate)fnread_json(')
        body = rest[:end]
        self.assertIn('letmutbits:BTreeSet<_>', body)
        self.assertIn('(2..=5).contains(b)', body)
        self.assertIn('bits.insert(ifbit==5{4}else{bit+1});', body)

    def test_single_width_checkpoints_pair_with_their_adjacent_tier(self):
        for observed, expected in (({2}, [2, 3]), ({3}, [3, 4]), ({4}, [4, 5]), ({5}, [4, 5])):
            with self.subTest(observed=observed):
                self.assertEqual(oracle.expected_decoder_family(observed), expected)

    def test_multi_width_checkpoints_form_the_family_the_loader_would_build(self):
        self.assertEqual(oracle.expected_decoder_family([3, 4]), [3, 4])
        self.assertEqual(oracle.expected_decoder_family([2, 3]), [2, 3])
        self.assertEqual(oracle.expected_decoder_family([2, 4]), [2, 4])
        self.assertEqual(oracle.expected_decoder_family([3, 3, 4, 4]), [3, 4])

    def test_only_widths_the_pin_can_encode_are_accepted(self):
        with self.assertRaises(ValueError) as caught:
            oracle.expected_decoder_family([6])
        self.assertIn('K2..K5', str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            oracle.expected_decoder_family([])
        self.assertIn('no trellis widths', str(caught.exception))


class SnapshotFixture:
    """Minimal metadata-only stand-ins for the manifests the loader reads."""

    @staticmethod
    def staged(root, storage, extra=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        manifest = {'tensor_storage': {
            f'layers.{i // 3}.ffn.experts.{i % 6}.{PROJECTIONS[i % 3]}':
                {'bits_per_weight': width} for i, width in enumerate(storage)}}
        manifest.update(extra or {})
        (root / 'quantize_config.json').write_text(json.dumps(manifest))
        return root

    @staticmethod
    def raw(root, quantization_config):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / 'config.json').write_text(
            json.dumps({'quantization_config': quantization_config}))
        return root


class FamilyRuleProvenanceTests(unittest.TestCase):
    """`python/tools/v41_exl3_family.py` is the ONE owner of the tier-family rule.

    The oracle and the offline tile bench consume it through import shims. A shim
    that quietly grew its own copy of the rule would keep passing every behavioural
    test while drifting from what the other consumer enforces, so identity is
    asserted here rather than trusted.
    """

    def shared(self):
        family = getattr(oracle, '_V41_EXL3_FAMILY', None)
        self.assertIsNotNone(family, 'the oracle must load the shared module, not copy it')
        return family

    def test_the_oracle_reexports_the_shared_rule_by_identity(self):
        """Same function objects, not look-alikes: one rule, two consumers."""
        family = self.shared()
        self.assertIs(oracle.expected_decoder_family, family.expected_decoder_family)
        self.assertIs(oracle.checkpoint_global_family, family.checkpoint_global_family)
        self.assertIs(oracle._staged_tensor_storage, family._staged_tensor_storage)
        self.assertEqual(family.__file__,
                         str(ROOT / 'python' / 'tools' / 'v41_exl3_family.py'))

    def test_the_oracle_does_not_reimplement_the_rule(self):
        source = ORACLE_SOURCE.read_text()
        for name in ('def expected_decoder_family', 'def checkpoint_global_family',
                     'def _staged_tensor_storage'):
            self.assertNotIn(name, source,
                             f'the oracle must import {name} from v41_exl3_family, '
                             f'not define its own copy')
        self.assertIn('v41_exl3_family.py', source, 'the shim must name its source')

    def test_the_tile_bench_consumes_the_same_rule(self):
        """The bench is a consumer, never a second authority."""
        bench = (ROOT / 'python' / 'tools' / 'bench_v41_exl3_tiles.py').read_text()
        self.assertIn('v41_exl3_family', bench,
                      'the bench must load the shared family module')
        for name in ('def expected_decoder_family', 'def checkpoint_global_family'):
            self.assertNotIn(name, bench, f'the bench redefines {name}')

    def test_the_shared_module_stays_dependency_free(self):
        """Pure stdlib by AST, or the oracle's CPU tests silently become GPU tests.

        Checked on import statements only: the word CUDA legitimately appears in the
        module's prose, which is not what this guards.
        """
        tree = ast.parse((ROOT / 'python' / 'tools' / 'v41_exl3_family.py').read_text())
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {alias.name.split('.')[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots.add(node.module.split('.')[0])
        self.assertTrue(roots <= {'json', 're', '__future__'},
                        f'unexpected import roots in the family module: {sorted(roots)}')


class CoverageGateTests(unittest.TestCase):
    """The coverage gate is its own decision, apart from family pairing.

    Pairing against the checkpoint-wide family is mandatory on every path; this gate
    is the separate question of whether a run demonstrates that the encoder reads
    per-projection tier membership.
    """

    def test_each_state_decides_the_gate(self):
        cases = (
            # (dspark_stage, uniform_global, mixed_sample, accepted, why)
            (None, False, False, False, 'neither: family uniformity absent, lookup unproven'),
            (None, True, False, True, 'checkpoint-wide single width: empty tier is the point'),
            (None, False, True, True, 'an expert mixes widths across its own projections'),
            (None, True, True, True, 'either proof suffices'),
            (0, False, False, True, 'sampled draft stage skips only the coverage gate'),
            (2, False, False, True, 'draft stages are legitimately uniform'),
        )
        for stage, uniform, mixed, accepted, why in cases:
            with self.subTest(dspark_stage=stage, uniform_global=uniform, mixed=mixed):
                self.assertEqual(oracle.coverage_gate_passed(stage, uniform, mixed),
                                 accepted, why)

    def test_gate_decision_is_independent_of_the_family_pairing(self):
        """Passing the coverage gate must never excuse a mismatched declaration."""
        self.assertTrue(oracle.coverage_gate_passed(None, True, False))
        with self.assertRaises(ValueError) as caught:
            oracle.assert_declared_family([3, 4], [4, 5], 47232, {'4': 47232})
        self.assertIn('forms the serving family [4, 5]', str(caught.exception))


class CheckpointWideFamilyTests(unittest.TestCase):
    """Pairing authority: the loader's family is a checkpoint property, not a sample one."""

    def test_manifest_family_and_distribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ('uniform K3', [3] * 24, ([3, 4], 24, True, {'3': 24})),
                ('uniform K4', [4] * 24, ([4, 5], 24, True, {'4': 24})),
                ('uniform K5 pairs down', [5] * 24, ([4, 5], 24, True, {'5': 24})),
                ('mixed 3/4', [3] * 18 + [4] * 6, ([3, 4], 24, False, {'3': 18, '4': 6})),
                ('non-adjacent 2/4', [2] * 18 + [4] * 6, ([2, 4], 24, False, {'2': 18, '4': 6})),
                ('three widths', [2] * 8 + [3] * 8 + [4] * 8,
                 ([2, 3, 4], 24, False, {'2': 8, '3': 8, '4': 8})),
            )
            for label, storage, expected in cases:
                with self.subTest(case=label):
                    snapshot = SnapshotFixture.staged(root / label.replace(' ', '_'), storage)
                    self.assertEqual(oracle.checkpoint_global_family(snapshot), expected)

    def test_staged_manifest_never_falls_back_to_top_level_bits(self):
        """A packaging default is not evidence of 47k per-projection widths."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, storage in (('absent', None), ('empty', {}), ('not a table', 3)):
                with self.subTest(case=label):
                    manifest = root / label
                    manifest.mkdir()
                    body = {'bits': 2, 'format': 'exl3'}
                    if storage is not None:
                        body['tensor_storage'] = storage
                    (manifest / 'quantize_config.json').write_text(json.dumps(body))
                    with self.assertRaises(ValueError) as caught:
                        oracle.checkpoint_global_family(manifest)
                    self.assertIn('nonempty tensor_storage', str(caught.exception))

    def test_a_holed_or_untyped_table_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = SnapshotFixture.staged(root / 'holed', [3, 4, 4])
            manifest = snapshot / 'quantize_config.json'
            table = json.loads(manifest.read_text())['tensor_storage']
            table[next(iter(table))].pop('bits_per_weight')
            manifest.write_text(json.dumps({'tensor_storage': table}))
            with self.assertRaises(ValueError) as caught:
                oracle.checkpoint_global_family(snapshot)
            self.assertIn('no bits_per_weight', str(caught.exception))

            snapshot = SnapshotFixture.staged(root / 'floaty', [3, 4])
            manifest = snapshot / 'quantize_config.json'
            table = json.loads(manifest.read_text())['tensor_storage']
            table[next(iter(table))]['bits_per_weight'] = 4.0
            manifest.write_text(json.dumps({'tensor_storage': table}))
            with self.assertRaises(ValueError) as caught:
                oracle.checkpoint_global_family(snapshot)
            self.assertIn('non-integer width', str(caught.exception))

    def test_raw_publication_mirrors_the_loader_gates(self):
        """parse_raw_publication insists on exl3 + mcg + mtp_experts=source + bits."""
        good = {'quant_method': 'exl3', 'codebook': 'mcg', 'mtp_experts': 'source', 'bits': 2}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = SnapshotFixture.raw(root / 'raw', good)
            self.assertEqual(oracle.checkpoint_global_family(snapshot), ([2, 3], 0, True, {}))
            for key, bad in (('quant_method', 'exllamav3'), ('codebook', 'none'),
                             ('mtp_experts', 'quantize'), ('bits', 9), ('bits', '3'),
                             ('bits', True)):
                with self.subTest(**{key: bad}):
                    broken = dict(good)
                    broken[key] = bad
                    path = root / f'{key}-{bad}'
                    snapshot = SnapshotFixture.raw(path, broken)
                    with self.assertRaises(ValueError) as caught:
                        oracle.checkpoint_global_family(snapshot)
                    self.assertIn(key, str(caught.exception))
            with self.assertRaises(ValueError) as caught:
                oracle.checkpoint_global_family(root / 'nothing-here')
            self.assertIn('neither tensor_storage nor a quantization_config',
                          str(caught.exception))

    def test_declaration_order_is_load_not_aesthetic(self):
        """`decoder_family` sorts; the runtime binary-searches and tags directories."""
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = SnapshotFixture.staged(Path(temporary) / 'case', [3] * 18 + [4] * 6)
            family, count, uniform, histogram = oracle.checkpoint_global_family(snapshot)
            oracle.assert_declared_family([3, 4], family, count, histogram)
            for wrong in ([4, 3], [3, 3], [3], [3, 4, 2]):
                with self.subTest(declared=wrong):
                    with self.assertRaises(ValueError) as caught:
                        oracle.assert_declared_family(wrong, family, count, histogram)
                    self.assertIn('export declares tiers', str(caught.exception))
            self.assertIn("'3': 18", str(histogram), 'the report must classify the spread')


class SampleProjectionContractTests(unittest.TestCase):
    """What a handful of sampled experts may and may not be used for."""

    def test_membership_follows_the_projection_that_declares_it(self):
        members, observed, mixed = oracle.sample_projection_contract(
            [3, 4], bitmaps([{ 'w1': 3, 'w3': 3, 'w2': 4},
                             {'w1': 4, 'w3': 4, 'w2': 3}] + [3] * 4), 6)
        self.assertEqual(observed, {3, 4})
        self.assertTrue(mixed)
        self.assertEqual(members[3]['w2'], (1, 2, 3, 4, 5))
        self.assertEqual(members[4]['w2'], (0,))
        self.assertEqual(members[3]['w1'], (0, 2, 3, 4, 5))

    def test_every_declared_tier_must_be_exercised_when_the_model_is_mixed(self):
        with self.assertRaises(ValueError) as caught:
            oracle.sample_projection_contract([3, 4], bitmaps([3] * 6), 6)
        self.assertIn('tier [4]', str(caught.exception))
        self.assertIn('--sample-experts', str(caught.exception))

    def test_a_uniform_checkpoint_may_leave_its_adjacent_tier_empty(self):
        members, observed, mixed = oracle.sample_projection_contract(
            [2, 3], bitmaps([2] * 6), 6, require_all_tiers=False)
        self.assertEqual(observed, {2})
        self.assertFalse(mixed)
        self.assertEqual(members[3]['w1'], (), 'the empty tier stays empty, as the loader pads it')
        self.assertEqual(members[2]['w1'], (0, 1, 2, 3, 4, 5))

    def test_a_sample_cannot_legitimise_the_wrong_family(self):
        """The reviewed hole, from the other side: sampled K4, K3 elsewhere.

        The sample contract has nothing to say about the family; the manifest sweep
        is what refuses, so this asserts the pair of them end to end.
        """
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = SnapshotFixture.staged(Path(temporary) / 'wide',
                                              [4] * 18 + [3] * 6)
            # All-K4 sample under a [4, 5] declaration: refused outright when the
            # tier must be exercised, and tolerated-but-overridden when it is not.
            with self.assertRaises(ValueError):
                oracle.sample_projection_contract([4, 5], bitmaps([4] * 6), 6)
            oracle.sample_projection_contract([4, 5], bitmaps([4] * 6), 6,
                                              require_all_tiers=False)
            family, count, uniform, histogram = oracle.checkpoint_global_family(snapshot)
            self.assertEqual((family, count, uniform), ([3, 4], 24, False))
            with self.assertRaises(ValueError) as caught:
                oracle.assert_declared_family([4, 5], family, count, histogram)
            self.assertIn('forms the serving family [3, 4]', str(caught.exception))

    def test_sample_cannot_contain_an_undeclared_width(self):
        with self.assertRaises(ValueError) as caught:
            oracle.sample_projection_contract([3, 4], bitmaps([3, 3, 3, 3, 3, 5]), 6,
                                             require_all_tiers=False)
        self.assertIn('declares no tier for', str(caught.exception))

    def test_missing_width_metadata_fails_closed_instead_of_keyerror(self):
        tables = bitmaps([3] * 6)
        del tables[(2, 'w2')]
        with self.assertRaises(ValueError) as caught:
            oracle.sample_projection_contract([3, 4], tables, 6)
        self.assertIn("(2, 'w2')", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            oracle.sample_projection_contract([3, 4], bitmaps([3] * 6), 7)
        self.assertIn('missing trellis width metadata', str(caught.exception))

    def test_non_integer_widths_are_refused(self):
        """A hand-edited v41_exl3.json must not coerce its way past the rule."""
        for widths in ([4.0] * 6, [True] * 6):
            with self.subTest(widths=widths):
                with self.assertRaises(ValueError) as caught:
                    oracle.expected_decoder_family(widths)
                self.assertIn('must be integers', str(caught.exception))
        self.assertEqual(oracle.expected_decoder_family([4, 5]), [4, 5])

    def test_each_part_is_classified_and_must_be_encodable(self):
        """Globally mixed does not mean every population is mixed."""
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / 'parts'
            storage = {}
            for index, width in enumerate([3] * 18 + [4] * 6):
                stem = PROJECTIONS[index % 3]
                storage[f'layers.{index // 3}.ffn.experts.{index % 6}.{stem}'] = \
                    {'bits_per_weight': width}
            for expert in range(9):
                storage[f'mtp.0.ffn.experts.{expert}.w1'] = {'bits_per_weight': 2}
            (snapshot.parent / 'parts').mkdir(parents=True, exist_ok=True)
            (snapshot / 'quantize_config.json').write_text(
                json.dumps({'tensor_storage': storage}))
            family, count, uniform, histogram = oracle.checkpoint_global_family(snapshot)
            self.assertEqual((family, count, uniform), ([2, 3, 4], 33, False))
            parts = oracle.checkpoint_part_families(snapshot)
            self.assertEqual(parts['target']['family'], [3, 4])
            self.assertEqual(parts['target']['count'], 24)
            self.assertEqual(parts['target']['histogram'], {'3': 18, '4': 6})
            self.assertFalse(parts['target']['uniform'])
            self.assertEqual(parts['mtp']['family'], [2, 3])
            self.assertTrue(parts['mtp']['uniform'], 'a uniform draft block in a mixed model')
            self.assertEqual(parts['mtp']['count'], 9)
            # Populations are counted per part: the target table here cycles six
            # experts, the draft block carries its own nine.
            self.assertEqual(parts['target']['experts'], 6)
            self.assertEqual(parts['mtp']['experts'], 9)
            oracle.assert_part_widths(parts['mtp'], 'mtp', [2, 3])
            with self.assertRaises(ValueError) as caught:
                oracle.assert_part_widths(parts['target'], 'target', [2, 3])
            self.assertIn('declares no tier for', str(caught.exception))
            # A part that does not exist is not a violation of the subset rule.
            oracle.assert_part_widths(None, 'mtp', [3, 4])

    def test_unrouted_entries_in_the_table_are_caught(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = SnapshotFixture.staged(Path(temporary) / 'extra', [3, 4])
            path = snapshot / 'quantize_config.json'
            table = json.loads(path.read_text())['tensor_storage']
            table['layers.0.attention.qkv'] = {'bits_per_weight': 4}
            path.write_text(json.dumps({'tensor_storage': table}))
            with self.assertRaises(ValueError) as caught:
                oracle.checkpoint_part_families(snapshot)
            self.assertIn('not routed projections', str(caught.exception))

    def test_tier_exercise_follows_the_sampled_part_not_the_whole_model(self):
        """"Globally mixed" must not force a uniform part, nor excuse a mixed one."""
        parts = {'target': {'uniform': True, 'count': 12, 'family': [3, 4],
                            'histogram': {'3': 12}, 'layers': [0]},
                 'mtp': {'uniform': False, 'count': 9, 'family': [3, 4],
                         'histogram': {'3': 6, '4': 3}, 'layers': [0]}}
        self.assertEqual(oracle.sampled_part_policy(parts, 'target', False),
                         (False, 'target', 'target part'))
        self.assertEqual(oracle.sampled_part_policy(parts, 'mtp', False),
                         (True, 'mtp', 'mtp part'))
        # No per-tensor table (raw publication): only the global flag is available.
        empty = {'target': None, 'mtp': None}
        self.assertEqual(oracle.sampled_part_policy(empty, 'target', True),
                         (False, 'target', 'checkpoint-wide (part not declared)'))
        self.assertEqual(oracle.sampled_part_policy(empty, 'target', False),
                         (True, 'target', 'checkpoint-wide (part not declared)'))
        with self.assertRaises(ValueError) as caught:
            oracle.sampled_part_policy(empty, 'mtp', True)
        self.assertIn('refusing to fall back to target layers', str(caught.exception))
        with self.assertRaises(ValueError):
            oracle.sampled_part_policy(parts, 'ple', False)

    def test_the_report_names_its_evidence_mode_and_tier_exercise(self):
        """A synthetic run must not be readable as checkpoint evidence."""
        source = ORACLE_SOURCE.read_text()
        self.assertIn("'evidence_mode': 'synthetic-fixture' if args.fixture is not None"
                      " else 'real-checkpoint'", source)
        self.assertIn("'tier_exercise'", source)
        self.assertIn("'sample_experts'", source)
        for key in ('tier_target_family', 'tier_target_histogram', 'tier_mtp_family',
                    'tier_mtp_uniform', 'sampled_part', 'require_all_tiers'):
            self.assertIn(f"'{key}'", source, f'report must publish {key}')
        # Both modes are decided by the caller, never inferred from the data.
        self.assertEqual(source.count("'evidence_mode'"), 1)

    def test_the_reference_compile_cannot_drift_from_the_declared_tiers(self):
        """The oracle compiles its own reference; the pin defaults to (3, 4).

        Nothing here can run CUDA on CPU, so the guard is textual: the compile call
        must pass both tier widths explicitly and echo the launch into the report.
        A default slipping back in would make every non-[3,4] family unreachable and
        fail only as an opaque storage-extent error.
        """
        source = ORACLE_SOURCE.read_text()
        compile_call = source[source.index('launch=compile_mixed_trellis('):]
        compile_call = compile_call[:compile_call.index('buffers=')]
        self.assertIn("tier0_bits=meta['bits'][0]", compile_call)
        self.assertIn("tier1_bits=meta['bits'][1]", compile_call)
        self.assertIn('launch.tier0_bits', source)
        self.assertIn('reference_tier_bits', source)


FAMILY_SOURCE = ROOT / 'python' / 'tools' / 'v41_exl3_family.py'


def load_shared_family():
    """The exact instance the oracle resolved when it loaded."""
    module = oracle._V41_EXL3_FAMILY
    assert Path(module.__file__).resolve() == FAMILY_SOURCE.resolve()
    return module


class SharedFamilyModuleTests(unittest.TestCase):
    """The oracle and the offline tile bench must consult ONE copy of the rule.

    The family helpers were extracted to `v41_exl3_family.py` (pure stdlib) so a
    rule edit lands everywhere at once; these tests pin that identity and the
    width type guard (bool is an int in Python and must not slip through).
    """

    def test_oracle_family_helpers_are_the_shared_module_itself(self):
        shared = load_shared_family()
        self.assertIs(oracle.expected_decoder_family, shared.expected_decoder_family)
        self.assertIs(oracle.checkpoint_global_family, shared.checkpoint_global_family)
        self.assertIs(oracle._staged_tensor_storage, shared._staged_tensor_storage)
        # And not a private second copy: the oracle file itself must not
        # re-define the extracted helpers.
        source = ORACLE_SOURCE.read_text()
        for name in ('def expected_decoder_family', 'def checkpoint_global_family',
                     'def _staged_tensor_storage'):
            self.assertNotIn(name, source)

    def test_shared_module_is_pure_stdlib(self):
        source = FAMILY_SOURCE.read_text()
        for banned in ('import torch', 'import ctypes', '_pinned_sparkinfer',
                       'PolicyContext', 'get_auto_policy'):
            self.assertNotIn(banned, source)

    def test_bool_is_refused_as_a_width_although_it_is_an_int(self):
        for observed in ([True], [3, False], (True, True)):
            with self.subTest(observed=observed):
                with self.assertRaises(ValueError) as caught:
                    oracle.expected_decoder_family(observed)
                self.assertIn('integers', str(caught.exception))

    def test_non_integer_width_types_are_refused(self):
        for observed in ([3.0], ['3'], [None]):
            with self.subTest(observed=observed):
                with self.assertRaises(ValueError):
                    oracle.expected_decoder_family(observed)


class SampleExpertBoundTests(unittest.TestCase):
    """A dspark sample is bounded by the sampled part's population, not by the
    384-target constant (mtp blocks carry their own, smaller expert count)."""

    PARTS = {'target': {'experts': 384}, 'mtp': {'experts': 128}}

    def test_each_population_bounds_its_own_sample(self):
        self.assertEqual(oracle.sample_expert_bound(self.PARTS, 'target'), 384)
        self.assertEqual(oracle.sample_expert_bound(self.PARTS, 'mtp'), 128)

    def test_raw_publications_are_bounded_by_the_config_population(self):
        raw = {'target': None, 'mtp': None}
        # The probe module's slot count (e.g. 6) is NOT a population: explicit
        # indices beyond it stay legal up to the config-declared 384 experts.
        self.assertEqual(oracle.sample_expert_bound(raw, 'target', 384), 384)
        sample = (6, 200, 383)
        self.assertEqual([e for e in sample if not 0 <= e < 384], [])
        self.assertEqual([e for e in (384,) if not 0 <= e < 384], [384])

    def test_a_raw_publication_without_any_declared_population_refuses(self):
        raw = {'target': None, 'mtp': None}
        for missing in (None, 0, -8):
            with self.subTest(raw_population=missing), self.assertRaises(ValueError):
                oracle.sample_expert_bound(raw, 'target', missing)

    def test_the_mtp_sample_index_validation_uses_the_bound(self):
        # Mirror of the main() guard: index 200 is target-legal but names no
        # mtp expert, and must refuse cleanly instead of KeyErrors deeper in.
        ceiling = oracle.sample_expert_bound(self.PARTS, 'mtp')
        sample = tuple(range(127, -1, -1)) + (200,)
        offending = [e for e in sample if not 0 <= e < ceiling]
        self.assertEqual(offending, [200])
