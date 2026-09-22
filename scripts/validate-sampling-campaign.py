#!/usr/bin/env python3
"""Validate a sampling-comparison campaign's raw reports before any median is published.

CPU-only, offline: reads the raw per-repeat reports produced by
``scripts/bench-ds41-release-decode.py`` and refuses to produce a result unless
the whole matrix is complete, self-consistent, quality-clean, identity-pinned and
actually interleaved by raw timestamp. It never contacts a server, GPU or
network, and it never edits the frozen harness or aggregator.

The aggregator is deliberately thin; this validator is the publication gate the
campaign relies on. It recomputes every published number from the raw per-sample
timings instead of trusting the report's own summary. Gates (hard unless noted):

* completeness  exactly ``profiles x repeats`` files, every (profile, repeat)
                exactly once, one repeat per file, one weight-0 counting
                diagnostic per file, ``include_counting`` true;
* cases         every selected weighted case present, timed and passed, with a
                positive completion-token count and identical case order;
* identity      model, base_url, tokenizer/corpus hashes, source revision,
                harness hash, nonce seed, repeat count and per-case weights equal
                across all files; a non-empty identity snapshot is mandatory and
                must be pinned by ``--expect-identity-sha``;
* identity body the original identity FILE's raw SHA-256 equals both the expected
                SHA and every ``provenance.identity_sha256``, and every embedded
                ``provenance.identity`` is DEEP-EQUAL to the parsed file (no
                canonical rehash, so whitespace cannot cause a false mismatch);
                the snapshot must name the actual 1 RTX + 4 Spark topology with
                dspark enabled and an immutable local image id plus source and
                version labels for every host (a registry digest is only required
                at release time);
* vector scope  canonical requested vector per profile label; exact stochastic
                effective vector and empty ``non_applied_fields``; greedy
                effective vector with ``non_applied_fields == ['top_k']``;
                identical within a profile across repeats; profiles stay distinct;
* per-sample    every weighted sample's own request body must carry the profile's
                sampling fields and seed (no seed for greedy, and no ``seed`` key
                at all) and the report model; no fixed length or forced EOS;
* seed          every stochastic profile has one shared EXPLICIT non-null seed;
                greedy carries none and says ``greedy-ignores-seed``;
* natural       ``decode_length_policy.fixed_decode_tokens`` is null everywhere
                and the corpus fixture is mandatory (``--corpus`` defaults to the
                frozen release corpus): every wired sample, counting included,
                must report ``request.max_tokens == requested_max_tokens ==
                corpus budget`` with ``min_tokens`` omitted/0 and ``ignore_eos``
                false; actual completion length is free to stop early at EOS and
                is never required to equal the budget;
* n=median      a profile median is emitted only when all ``repeats`` values are
                present; every number is recomputed from raw samples;
* rotation      repeats are contiguous in raw ``started_ns`` order and each
                repeat block is the TRUE CYCLIC rotation of the profile order at
                offset ``repeat - 1`` (offsets 0/1/2), not merely a different
                order; profile-major ordering fails;
* quality       every weighted sample passed and every report is
                weighted-publishable;
* summary       per-repeat ``weighted_cases``, ``timed_tokens``,
                ``timed_seconds`` and ``weighted_observed_decode_tokens_per_second``
                are recomputed from the raw samples and compared against the
                report's own summary;
* diagnostics   weight-0 samples are excluded from the weighted median and their
                pass state is reported separately and prominently (soft gate: a
                diagnostic failure never blocks the weighted result but is always
                disclosed);
* lengths       per-case and per-profile completion-token spreads are recorded so
                the workload difference between profiles is stated, not hidden.

Expected identity file shape (extra keys are allowed and recorded)::

    {
      "model": {"id": "...", "revision": "..."},
      "topology": {"rtx_coordinators": 1, "spark_workers": 4, "dspark": true},
      "coordinator": {"host": "...", "gpus": [{"index": 0, "name": "...", "uuid": "..."}],
                      "image": {"id": "sha256:...", "source": "...", "version": "..."}},
      "spark_workers": [{"host": "...",
                         "image": {"id": "sha256:...", "source": "...", "version": "..."}}, ...]
    }
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path


DEFAULT_PROFILES = (
    'greedy',
    'temp0.2-topp0.95',
    'temp0.7-topp0.9',
    'temp0.7-minp0.05',
    'temp0.7-topk40',
)

# Canonical requested vectors, mirrored from the harness contract. A report whose
# profile label does not match its own requested vector is mislabelled or
# corrupted and must never be aggregated.
CANONICAL_VECTORS = {
    'greedy': {'temperature': 0.0, 'top_p': 1.0, 'top_k': 0, 'min_p': 0.0},
    'temp0.2-topp0.95': {'temperature': 0.2, 'top_p': 0.95, 'top_k': 0, 'min_p': 0.0},
    'temp0.7-topp0.9': {'temperature': 0.7, 'top_p': 0.9, 'top_k': 0, 'min_p': 0.0},
    'temp0.7-minp0.05': {'temperature': 0.7, 'top_p': 1.0, 'top_k': 0, 'min_p': 0.05},
    'temp0.7-topk40': {'temperature': 0.7, 'top_p': 1.0, 'top_k': 40, 'min_p': 0.0},
}
GREEDY_EFFECTIVE = {'temperature': 0.0, 'top_p': 1.0, 'top_k': 1, 'min_p': 0.0}
SAMPLING_KEYS = ('temperature', 'top_p', 'top_k', 'min_p')


def load_reports(paths):
    documents = []
    for path in paths:
        raw = Path(path).read_bytes()
        documents.append((json.loads(raw), hashlib.sha256(raw).hexdigest(), str(path)))
    return documents


def _weighted(samples):
    return [s for s in samples if s.get('weight', 0) > 0]


def _diagnostics(samples):
    return [s for s in samples if s.get('weight', 0) <= 0]


def _close(left, right, rel=1e-9):
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) <= rel * max(1.0, abs(left), abs(right))


def weighted_metric_from_raw(samples):
    """Recompute the weighted decode metric from the raw per-sample timings."""
    weighted = [s for s in samples
                if s.get('weight', 0) > 0
                and isinstance((s.get('usage') or {}).get('completion_tokens'), int)
                and 'finish_seconds' in s and 'first_output_seconds' in s]
    timed_tokens = sum(s['weight'] * (s['usage']['completion_tokens'] - 1) for s in weighted)
    timed_seconds = sum(s['weight'] * (s['finish_seconds'] - s['first_output_seconds'])
                        for s in weighted)
    return {
        'weighted_cases': len(weighted),
        'timed_tokens': timed_tokens,
        'timed_seconds': timed_seconds,
        'weighted_observed_decode_tokens_per_second': (
            timed_tokens / timed_seconds if timed_seconds > 0 else None),
    }


def _identity(document):
    provenance = document.get('provenance') or {}
    return {
        'model': document.get('model'),
        'base_url': document.get('base_url'),
        'tokenizer_sha256': document.get('tokenizer_sha256'),
        'corpus_sha256': document.get('corpus_sha256'),
        'identity_sha256': provenance.get('identity_sha256'),
        'source_revision': provenance.get('source_revision'),
        'harness_sha256': (provenance.get('harness') or {}).get('sha256'),
        'selected_cases': sorted(document.get('selected_cases') or []),
        'nonce_seed': document.get('nonce_seed'),
        'repeats': document.get('repeats'),
        'encoding': (document.get('sampling') or {}).get('encoding'),
    }


def _case_weights(document):
    return {s.get('case'): s.get('weight') for s in _weighted(document.get('samples', []))}


def _vector_scope(document):
    sampling = document.get('sampling') or {}
    return {
        'requested_vector': sampling.get('requested_vector'),
        'effective_vector': sampling.get('effective_vector'),
        'request_fields': sampling.get('request_fields'),
        'non_applied_fields': sampling.get('non_applied_fields'),
        'seed': sampling.get('seed'),
        'seed_source': sampling.get('seed_source'),
        'greedy': sampling.get('greedy'),
        'decode_length_policy': document.get('decode_length_policy'),
    }


def _start_ns(document):
    values = [s.get('started_ns') for s in document.get('samples', [])
              if isinstance(s.get('started_ns'), int)]
    return min(values) if values else None


def _repeat_blocks(sequence):
    """Group an ordered (profile, repeat) sequence into contiguous repeat blocks."""
    blocks = []
    for profile, repeat in sequence:
        if not blocks or blocks[-1][0] != repeat:
            blocks.append((repeat, [profile]))
        else:
            blocks[-1][1].append(profile)
    return blocks


def _sequence_grouped_by(sequence, index):
    """True when the sequence is grouped by `sequence[index]` (each value one run)."""
    seen = set()
    current = None
    for entry in sequence:
        value = entry[index]
        if value != current:
            if value in seen:
                return False
            seen.add(value)
            current = value
    return True


def _image_failures(where, image):
    failures = []
    if not isinstance(image, dict):
        return [f"{where}.image is missing (immutable local image identity required)"]
    if not image.get('id'):
        failures.append(f"{where}.image.id is missing (immutable local image id)")
    for field in ('source', 'version'):
        if not image.get(field):
            failures.append(f"{where}.image.{field} is missing")
    return failures


def identity_schema_failures(identity):
    """Require a non-empty identity naming the real 1 RTX + 4 Spark dspark topology."""
    if not isinstance(identity, dict) or not identity:
        return ["identity snapshot is empty"]
    failures = []
    topology = identity.get('topology')
    if not isinstance(topology, dict):
        failures.append("identity.topology is missing")
        topology = {}
    if topology.get('rtx_coordinators') != 1:
        failures.append(f"identity.topology.rtx_coordinators must be 1, "
                        f"got {topology.get('rtx_coordinators')!r}")
    if topology.get('spark_workers') != 4:
        failures.append(f"identity.topology.spark_workers must be 4, "
                        f"got {topology.get('spark_workers')!r}")
    if topology.get('dspark') is not True:
        failures.append(f"identity.topology.dspark must be true, got {topology.get('dspark')!r}")
    model = identity.get('model')
    if not isinstance(model, dict) or not model.get('id') or not model.get('revision'):
        failures.append("identity.model.id and identity.model.revision are required")
    coordinator = identity.get('coordinator')
    if not isinstance(coordinator, dict):
        failures.append("identity.coordinator is missing")
    else:
        gpus = coordinator.get('gpus')
        if not isinstance(gpus, list) or len(gpus) != 1:
            failures.append("identity.coordinator.gpus must list exactly one RTX")
        if not coordinator.get('host'):
            failures.append("identity.coordinator.host is missing")
        failures.extend(_image_failures('identity.coordinator', coordinator.get('image')))
    workers = identity.get('spark_workers')
    if not isinstance(workers, list) or len(workers) != 4:
        failures.append("identity.spark_workers must list exactly four Spark hosts")
    else:
        for index, worker in enumerate(workers):
            if not isinstance(worker, dict) or not worker.get('host'):
                failures.append(f"identity.spark_workers[{index}].host is missing")
            failures.extend(_image_failures(f"identity.spark_workers[{index}]",
                                            (worker or {}).get('image')))
    return failures


def _corpus_budget(corpus, case):
    """Natural output budget for a case id (weighted case, counting or orchid)."""
    if not isinstance(corpus, dict):
        return None
    if case in (corpus.get('cases') or {}):
        return corpus['cases'][case].get('max_tokens')
    if case in ('counting', 'orchid'):
        return (corpus.get(case) or {}).get('max_tokens')
    return None


def _sample_policy_failures(key, document, scope, sample, corpus):
    """Every sample (weighted and diagnostic) must use the profile's natural policy."""
    case = sample.get('case')
    failures = []
    request = sample.get('request')
    if not isinstance(request, dict):
        return [f"{key}/{case}: sample has no request body"]
    fields = {name: request.get(name) for name in SAMPLING_KEYS if name in request}
    if fields != scope.get('request_fields'):
        failures.append(f"{key}/{case}: sample request sampling {fields} != profile "
                        f"{scope.get('request_fields')}")
    if request.get('model') != document.get('model'):
        failures.append(f"{key}/{case}: sample request model {request.get('model')!r} != "
                        f"report model {document.get('model')!r}")
    if scope.get('greedy'):
        if 'seed' in request:
            failures.append(f"{key}/{case}: greedy sample request must omit the seed")
    elif request.get('seed') != scope.get('seed'):
        failures.append(f"{key}/{case}: sample request seed {request.get('seed')!r} != "
                        f"profile seed {scope.get('seed')!r}")
    expected_max = _corpus_budget(corpus, case)
    if expected_max is not None:
        if sample.get('requested_max_tokens') != expected_max:
            failures.append(f"{key}/{case}: requested_max_tokens "
                            f"{sample.get('requested_max_tokens')!r} != corpus budget {expected_max}")
        if request.get('max_tokens') != expected_max:
            failures.append(f"{key}/{case}: request.max_tokens {request.get('max_tokens')!r} != "
                            f"corpus budget {expected_max}")
    if sample.get('requested_decode_tokens') is not None:
        failures.append(f"{key}/{case}: sample pins a fixed decode length")
    if sample.get('requested_ignore_eos') or 'ignore_eos' in request:
        failures.append(f"{key}/{case}: sample forces ignore_eos")
    if sample.get('requested_min_tokens') not in (None, 0) or request.get('min_tokens'):
        failures.append(f"{key}/{case}: sample sets min_tokens "
                        f"{sample.get('requested_min_tokens')!r}/{request.get('min_tokens')!r}; "
                        f"natural budgets require it omitted/0")
    return failures


def validate(documents, profiles=DEFAULT_PROFILES, repeats=3, identity_required=True,
             identity_document=None, identity_sha256=None, expect_identity_sha=None,
             corpus=None, corpus_sha256=None):
    """Return a validation record; `passed` is True only when no hard gate failed."""
    failures = []
    by_key = {}
    for document, sha256, path in documents:
        profile = (document.get('sampling') or {}).get('profile')
        summaries = document.get('repeat_summaries') or []
        if profile not in profiles:
            failures.append(f"{path}: unknown profile {profile!r}")
            continue
        if len(summaries) != 1:
            failures.append(f"{path}: expected exactly one repeat summary "
                            f"(run one repeat per report via --repeat-index), got {len(summaries)}")
            continue
        repeat = summaries[0].get('repeat')
        key = (profile, repeat)
        if key in by_key:
            failures.append(f"duplicate report for {key}: {by_key[key][2]} and {path}")
            continue
        by_key[key] = (document, sha256, path)

    expected = {(profile, repeat) for profile in profiles for repeat in range(1, repeats + 1)}
    missing = sorted(expected - set(by_key))
    extra = sorted(set(by_key) - expected)
    if len(documents) != len(profiles) * repeats:
        failures.append(f"expected {len(profiles) * repeats} raw reports, got {len(documents)}")
    if missing:
        failures.append(f"missing (profile, repeat) reports: {missing}")
    if extra:
        failures.append(f"unexpected (profile, repeat) reports: {extra}")

    # Identity across every file, plus the mandatory pinned snapshot.
    identities = {key: _identity(document) for key, (document, _, _) in by_key.items()}
    if identities:
        reference_key = sorted(identities)[0]
        reference = identities[reference_key]
        for key, identity in sorted(identities.items()):
            if identity != reference:
                differing = {field: (reference[field], identity[field])
                             for field in reference if identity[field] != reference[field]}
                failures.append(f"{key}: cross-file identity differs from {reference_key}: {differing}")
        if identity_required and reference.get('encoding') != 'strict':
            failures.append(f"sampling.encoding is {reference.get('encoding')!r}, expected 'strict'")
    if identity_required:
        if identity_document is None:
            failures.append("an identity snapshot file is required for a release campaign")
        if identity_sha256 is None:
            failures.append("the identity file's raw SHA-256 could not be computed")
        if not expect_identity_sha:
            failures.append("--expect-identity-sha is required; the campaign must pin the identity hash")
        elif identity_sha256 is not None and identity_sha256 != expect_identity_sha:
            failures.append(f"identity file raw SHA-256 {identity_sha256} != expected {expect_identity_sha}")
        if identity_sha256 is not None:
            for key, identity in sorted(identities.items()):
                if identity.get('identity_sha256') != identity_sha256:
                    failures.append(f"{key}: provenance.identity_sha256 != identity file SHA-256")
        if identity_document is not None:
            failures.extend(identity_schema_failures(identity_document))
            for key in sorted(by_key):
                embedded = (by_key[key][0].get('provenance') or {}).get('identity')
                if embedded != identity_document:
                    failures.append(f"{key}: embedded provenance.identity is not deep-equal to the "
                                    f"identity file")
            identity_model = (identity_document.get('model') or {}).get('id')
            if identity_model:
                for key in sorted(by_key):
                    if by_key[key][0].get('model') != identity_model:
                        failures.append(f"{key}: report model {by_key[key][0].get('model')!r} != "
                                        f"identity.model.id {identity_model!r}")

    if corpus is None:
        failures.append("a corpus fixture is required to gate natural budgets and weights")

    weights = {key: _case_weights(document) for key, (document, _, _) in by_key.items()}
    if weights:
        reference_key = sorted(weights)[0]
        for key, value in sorted(weights.items()):
            if value != weights[reference_key]:
                failures.append(f"{key}: per-case weights differ from {reference_key}")
    if corpus is not None:
        expected_weights = {case: corpus['cases'][case]['weight']
                            for case in corpus['weighted_case_ids']}
        expected_cases = sorted(corpus['weighted_case_ids'])
        for key, value in sorted(weights.items()):
            if value != expected_weights:
                failures.append(f"{key}: per-case weights differ from the corpus fixture")
        reference = identities.get(sorted(identities)[0]) if identities else {}
        if corpus_sha256 is not None and reference.get('corpus_sha256') != corpus_sha256:
            failures.append("corpus_sha256 does not match the provided corpus fixture")
        for key, identity in sorted(identities.items()):
            if identity.get('selected_cases') != expected_cases:
                failures.append(f"{key}: selected_cases != corpus weighted_case_ids")

    # Vector scope within a profile; profiles must stay distinct.
    scopes = {key: _vector_scope(document) for key, (document, _, _) in by_key.items()}
    for profile in profiles:
        profile_keys = sorted(key for key in scopes if key[0] == profile)
        if len(profile_keys) < 2:
            continue
        reference_key = profile_keys[0]
        for key in profile_keys[1:]:
            if scopes[key] != scopes[reference_key]:
                differing = {field: (scopes[reference_key][field], scopes[key][field])
                             for field in scopes[reference_key]
                             if scopes[key][field] != scopes[reference_key][field]}
                failures.append(f"{key}: within-profile vector/policy differs from "
                                f"{reference_key}: {differing}")
    distinct_vectors = {json.dumps(value.get('request_fields'), sort_keys=True)
                        for value in scopes.values()}
    if len(scopes) > 1 and len(distinct_vectors) < 2:
        failures.append("every profile sends the same request fields; profiles are not distinct")

    # Exact canonical binding: a profile label must match its own vector, not just
    # differ from the other profiles.
    for key in sorted(scopes):
        canonical = CANONICAL_VECTORS.get(key[0])
        if canonical is None:
            continue
        scope = scopes[key]
        requested = scope.get('requested_vector')
        fields = scope.get('request_fields')
        effective = scope.get('effective_vector')
        non_applied = scope.get('non_applied_fields')
        if requested != canonical:
            failures.append(f"{key}: requested_vector {requested} != canonical {canonical}")
        if key[0] == 'greedy':
            if fields != {'temperature': 0}:
                failures.append(f"{key}: greedy request_fields {fields} != {{'temperature': 0}}")
            if effective != GREEDY_EFFECTIVE:
                failures.append(f"{key}: greedy effective_vector {effective} != {GREEDY_EFFECTIVE}")
            if non_applied != ['top_k']:
                failures.append(f"{key}: greedy non_applied_fields {non_applied} != ['top_k']")
        else:
            if fields != canonical:
                failures.append(f"{key}: request_fields {fields} != canonical {canonical}")
            if effective != canonical:
                failures.append(f"{key}: effective_vector {effective} != canonical {canonical}")
            if non_applied != []:
                failures.append(f"{key}: stochastic non_applied_fields {non_applied} != []")

    # Case order must follow selected_cases for weighted samples and be identical
    # across every report (diagnostics included).
    observed_orders = {}
    for key in sorted(by_key):
        document = by_key[key][0]
        samples = document.get('samples', [])
        observed_orders[key] = [s.get('case') for s in samples]
        weighted_order = [s.get('case') for s in samples if s.get('weight', 0) > 0]
        selected_order = list(document.get('selected_cases') or [])
        if weighted_order != selected_order:
            failures.append(f"{key}: weighted case order {weighted_order} != selected_cases "
                            f"order {selected_order}")
    if len({tuple(order) for order in observed_orders.values()}) > 1:
        failures.append("sample case order differs across reports")

    # Seed rules: one shared explicit non-null seed for stochastic profiles, none
    # for greedy.
    greedy = {key[0]: (scopes[key] or {}).get('greedy') for key in scopes}
    stochastic_seeds = {scopes[key].get('seed') for key in scopes if not greedy.get(key[0])}
    if len(stochastic_seeds) > 1:
        failures.append(f"stochastic profiles do not share one seed: {sorted(map(str, stochastic_seeds))}")
    elif stochastic_seeds and None in stochastic_seeds:
        failures.append("stochastic profiles must record an explicit non-null seed")
    for key in scopes:
        if greedy.get(key[0]):
            if scopes[key].get('seed') is not None:
                failures.append(f"{key}: greedy profile must not carry a seed")
            if scopes[key].get('seed_source') != 'greedy-ignores-seed':
                failures.append(f"{key}: greedy seed_source should be 'greedy-ignores-seed', "
                                f"got {scopes[key].get('seed_source')!r}")
        elif scopes[key].get('seed_source') != 'explicit':
            failures.append(f"{key}: stochastic seed_source should be 'explicit', got "
                            f"{scopes[key].get('seed_source')!r}")

    # Natural budgets: no fixed length anywhere.
    for key in sorted(by_key):
        policy = by_key[key][0].get('decode_length_policy') or {}
        if policy.get('fixed_decode_tokens') is not None:
            failures.append(f"{key}: decode_length_policy.fixed_decode_tokens must be null "
                            f"(natural per-category budgets)")

    # Per (profile, repeat) completeness, quality, per-sample requests, diagnostics
    # and recomputed summary.
    length_rows = {}
    recomputed_rates = {}
    for key in sorted(by_key):
        document, _, path = by_key[key]
        samples = document.get('samples', [])
        weighted = _weighted(samples)
        selected = set(document.get('selected_cases') or [])
        cases = {s.get('case') for s in weighted}
        if cases != selected:
            failures.append(f"{key}: weighted cases {sorted(cases)} != selected {sorted(selected)}")
        if _start_ns(document) is None:
            failures.append(f"{key}: no sample carries an integer started_ns; rotation cannot be verified")
        if document.get('include_counting') is not True:
            failures.append(f"{key}: include_counting is not True")
        diagnostics = _diagnostics(samples)
        if len(diagnostics) != 1 or diagnostics[0].get('case') != 'counting':
            failures.append(f"{key}: expected exactly one weight-0 counting diagnostic, got "
                            f"{[s.get('case') for s in diagnostics]}")
        # The natural policy is checked on EVERY wired sample, counting included.
        for sample in samples:
            failures.extend(_sample_policy_failures(key, document, scopes[key], sample, corpus))
        for sample in weighted:
            case = sample.get('case')
            if 'finish_seconds' not in sample or 'first_output_seconds' not in sample:
                failures.append(f"{key}/{case}: weighted sample has no timing")
                continue
            tokens = (sample.get('usage') or {}).get('completion_tokens')
            if not isinstance(tokens, int) or tokens < 1:
                failures.append(f"{key}/{case}: completion_tokens={tokens!r} is not a positive count")
            length_rows.setdefault((key[0], case), []).append(tokens)
        failed = sorted(s.get('case') for s in weighted if s.get('passed') is not True)
        if failed:
            failures.append(f"{key}: weighted samples not passed: {failed}")
        if document.get('weighted_passed') is not True:
            failures.append(f"{key}: report.weighted_passed is not True")
        if document.get('publishable') is not True:
            failures.append(f"{key}: report.publishable is not True")
        summary = document['repeat_summaries'][0]
        recomputed = weighted_metric_from_raw(samples)
        if summary.get('weighted_cases') != recomputed['weighted_cases']:
            failures.append(f"{key}: summary weighted_cases {summary.get('weighted_cases')!r} != "
                            f"recomputed {recomputed['weighted_cases']}")
        for field in ('timed_tokens', 'timed_seconds',
                      'weighted_observed_decode_tokens_per_second'):
            if not _close(summary.get(field), recomputed[field]):
                failures.append(f"{key}: summary {field} {summary.get(field)!r} != recomputed "
                                f"{recomputed[field]!r}")
        recomputed_rates[key] = recomputed['weighted_observed_decode_tokens_per_second']

    # n-repeat medians from the RECOMPUTED rates, only when every repeat is present.
    medians = {}
    series = {}
    for profile in profiles:
        values = [recomputed_rates.get((profile, repeat)) for repeat in range(1, repeats + 1)]
        series[profile] = values
        if len(values) == repeats and all(value is not None for value in values):
            medians[profile] = statistics.median(values)
        else:
            failures.append(f"{profile}: cannot publish an n={repeats} median, values={values}")

    # Actual rotation from raw timestamps: contiguous repeat blocks that are the
    # true cyclic rotation at offset repeat-1.
    ordered = sorted(((_start_ns(document), key)
                      for key, (document, _, _) in by_key.items()
                      if _start_ns(document) is not None))
    sequence = [key for _, key in ordered]
    blocks = _repeat_blocks(sequence)
    profile_major = _sequence_grouped_by(sequence, 0)
    repeat_major = _sequence_grouped_by(sequence, 1)
    if len(profiles) > 1 and repeats > 1 and profile_major:
        failures.append("campaign is profile-major (all repeats of one profile together); "
                        "the required interleaved rotation did not happen")
    if not repeat_major:
        failures.append(f"repeats are not contiguous in raw started_ns order: "
                        f"{[repeat for repeat, _ in blocks]}")
    for repeat, block_profiles in blocks:
        if sorted(block_profiles) != sorted(profiles):
            failures.append(f"repeat {repeat} block has profiles {sorted(block_profiles)}, "
                            f"expected {sorted(profiles)}")
            continue
        offset = (repeat - 1) % len(profiles)
        expected_order = list(profiles[offset:]) + list(profiles[:offset])
        if block_profiles != expected_order:
            failures.append(f"repeat {repeat} order {block_profiles} != cyclic rotation "
                            f"offset {offset}: {expected_order}")
    rotated = len({tuple(block_profiles) for _, block_profiles in blocks}) > 1
    if len(profiles) > 1 and repeats > 1 and not rotated:
        failures.append("campaign rotation did not happen: every repeat used the same profile "
                        "order, so drift is not balanced across profiles")

    # Diagnostics: disclosed separately, never folded into the weighted median.
    diagnostics_summary = {}
    for profile in profiles:
        rows = []
        for repeat in range(1, repeats + 1):
            entry = by_key.get((profile, repeat))
            if entry is not None:
                rows.extend(_diagnostics(entry[0].get('samples', [])))
        diagnostics_summary[profile] = {
            'samples': len(rows),
            'cases': sorted({s.get('case') for s in rows}),
            'failed_cases': sorted({s.get('case') for s in rows if s.get('passed') is not True}),
            'all_passed': all(s.get('passed') is True for s in rows) if rows else True,
        }

    # Length spreads: state the workload difference instead of hiding it.
    length_spreads = {}
    for profile in profiles:
        cases = {}
        for case in sorted({case for (owner, case) in length_rows if owner == profile}):
            values = [value for value in length_rows[(profile, case)] if isinstance(value, int)]
            if values:
                cases[case] = {'samples': len(values), 'min': min(values),
                               'median': statistics.median(values), 'max': max(values)}
        length_spreads[profile] = cases

    return {
        'schema': 1,
        'profiles': list(profiles),
        'repeats': repeats,
        'files_expected': len(profiles) * repeats,
        'files_seen': len(documents),
        'passed': not failures,
        'failures': failures,
        'medians': medians,
        'repeat_series': series,
        'recomputed_rates': {'%s/r%s' % key: value
                             for key, value in sorted(recomputed_rates.items())},
        'diagnostics': diagnostics_summary,
        'length_spreads': length_spreads,
        'identity': {
            'file_sha256': identity_sha256,
            'expected_sha256': expect_identity_sha,
            'deep_equal': identity_document is not None and all(
                (by_key[key][0].get('provenance') or {}).get('identity') == identity_document
                for key in by_key) and bool(by_key),
            'schema_failures': identity_schema_failures(identity_document)
            if identity_document is not None else [],
        },
        'rotation': {
            'blocks': [{'repeat': repeat, 'profiles': block} for repeat, block in blocks],
            'profile_major': profile_major,
            'repeats_contiguous': repeat_major,
            'rotated': rotated,
            'observed_order': [[profile, repeat] for profile, repeat in sequence],
        },
        'inputs': [{'path': path, 'sha256': sha256} for _, sha256, path in documents],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--identity-file', type=Path, required=True,
                        help='original identity snapshot captured before the campaign')
    parser.add_argument('--expect-identity-sha', required=True,
                        help='expected raw SHA-256 of the identity file')
    parser.add_argument('--profiles', nargs='+', default=list(DEFAULT_PROFILES))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='sampling-campaign')
    parser.add_argument('--corpus', type=Path,
                        default=Path(__file__).with_name('fixtures') / 'release-semantic-corpus.json',
                        help='corpus fixture (default: the frozen release corpus); its SHA-256, '
                             'weights and natural budgets must match the reports')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    if not args.identity_file.is_file():
        parser.error(f'identity file not found: {args.identity_file}')
    identity_raw = args.identity_file.read_bytes()
    identity_document = json.loads(identity_raw)
    identity_sha256 = hashlib.sha256(identity_raw).hexdigest()
    if not args.corpus.is_file():
        parser.error(f'corpus fixture not found: {args.corpus}')
    corpus_raw = args.corpus.read_bytes()
    corpus_document = json.loads(corpus_raw)
    corpus_sha256 = hashlib.sha256(corpus_raw).hexdigest()
    documents = load_reports(args.reports)
    result = validate(documents, tuple(args.profiles), args.repeats,
                      identity_document=identity_document, identity_sha256=identity_sha256,
                      expect_identity_sha=args.expect_identity_sha,
                      corpus=corpus_document, corpus_sha256=corpus_sha256)
    result['label'] = args.label
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for profile in result['profiles']:
        median = result['medians'].get(profile)
        diagnostics = result['diagnostics'][profile]
        print(f"{profile}: median={median} weighted_passed={profile in result['medians']} "
              f"diagnostics_all_passed={diagnostics['all_passed']}")
    print(f"passed={result['passed']} failures={len(result['failures'])} "
          f"rotated={result['rotation']['rotated']}")
    if not result['passed']:
        for failure in result['failures']:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
