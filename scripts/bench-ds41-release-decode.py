#!/usr/bin/env python3
"""Run the nine-category and counting release decode workloads.

Sampling comparison support
---------------------------
This harness is the release headline decode battery: nine weighted content
categories plus optional counting. By default it measures the historical
`greedy` control (`temperature=0`) exactly as before, and every report records
the content categories, weights and decode-only metric unchanged.

It can now also drive a *matched* sampling comparison across five named
profiles. A profile is the complete request-level sampling vector, not just the
one filter under test: unrelated filters are explicitly disabled so a profile
cannot silently inherit a server default.

Canonical (vLLM) profile definitions:

  greedy            temperature=0.0  top_p=1.0   top_k=0 (disabled)  min_p=0.0
  temp0.2-topp0.95  temperature=0.2  top_p=0.95  top_k=0 (disabled)  min_p=0.0
  temp0.7-topp0.9   temperature=0.7  top_p=0.9   top_k=0 (disabled)  min_p=0.0
  temp0.7-minp0.05  temperature=0.7  top_p=1.0   top_k=0 (disabled)  min_p=0.05
  temp0.7-topk40    temperature=0.7  top_p=1.0   top_k=40            min_p=0.0

Semantics these encodings rely on:
  * Filter disabling follows vLLM conventions: `top_k=0` (or `-1`) disables
    top-k, `top_p=1.0` disables nucleus truncation, `min_p=0.0` disables min-p.
  * This engine's greedy rule is `temperature < 1e-5` OR `top_k == 1`; the vector
    it applies on the greedy path is
    `temperature=0.0, top_p=1.0, top_k=1, min_p=0.0`.
  * The host stochastic sampler order is mask -> temperature -> min_p -> top_k ->
    top_p; vLLM v1 uses temperature -> min_p -> top_k -> top_p.

Requested vs applied
--------------------
The harness is strict-only: it sends the canonical vector, including `top_k=0`
and `min_p`, and therefore requires an engine that implements those values.
There is no fallback encoding. A vector the engine cannot express is a hard
request error (HTTP 400), never a silent substitution. The report records both
the requested vector and the exact fields sent, and never claims a profile was
honoured merely because the request was accepted.

Stochastic profiles require an explicit `--seed`, because the server otherwise
generates a time-based seed per request and the run is not reproducible. An
unseeded stochastic run is refused unless `--unseeded` is passed, which records
the run as non-reproducible.

Decode-length policy
--------------------
The headline metric is decode-only and weight-normalized:

  sum(weight * (completion_tokens - 1)) / sum(weight * (finish - first_output))

Sampling changes which tokens are emitted, so output *length* becomes a
workload variable that must be separated from sampler cost. Where the native
API supports it, `--fixed-decode-tokens N` pins every sample to the same
generated length (`ignore_eos=true`, `min_tokens=N`, `max_tokens=N`) so the
timed token count is identical across profiles. When the server does not honour
the fixed length, the harness records the actual `completion_tokens` per sample
and the `fixed_length_honored` verdict instead of pretending the request was
met. EOS/stop behaviour (`ignore_eos`, `min_tokens`, `finish_reason`) is part
of the recorded metadata either way.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
import time

from tokenizers import Tokenizer


# Canonical vLLM-semantics sampling vectors. Every profile disables the filters
# it does not test: top_k=0 disables top-k, top_p=1.0 disables nucleus, and
# min_p=0.0 disables min-p.
SAMPLING_PROFILES = {
    'greedy': dict(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0),
    'temp0.2-topp0.95': dict(temperature=0.2, top_p=0.95, top_k=0, min_p=0.0),
    'temp0.7-topp0.9': dict(temperature=0.7, top_p=0.9, top_k=0, min_p=0.0),
    'temp0.7-minp0.05': dict(temperature=0.7, top_p=1.0, top_k=0, min_p=0.05),
    'temp0.7-topk40': dict(temperature=0.7, top_p=1.0, top_k=40, min_p=0.0),
}
DEFAULT_SAMPLING_PROFILE = 'greedy'

# Server defaults that apply when a sampling field is omitted. Recorded so a
# partially specified request can never be mistaken for a fully specified one.
SERVER_DEFAULTS_IF_UNSET = {
    'temperature': '0.0 (greedy)',
    'top_p': '0.95 when non-greedy',
    'top_k': '50 when non-greedy, capped at 64',
    'min_p': 'sent by the strict harness; the v11 candidate engine implements it',
    'seed': 'time-based generated value when unset',
}
SAMPLING_SEMANTICS = (
    'top_k=0 or -1 disables top-k; top_p=1.0 disables nucleus; min_p=0.0 disables '
    'min-p; greedy is temperature < 1e-5 or top_k == 1, and the applied greedy '
    'vector is temperature=0.0, top_p=1.0, top_k=1, min_p=0.0; host order is '
    'mask -> temperature -> min_p -> top_k -> top_p '
    '(temperature -> min_p -> top_k -> top_p in vLLM v1)'
)

# dSpark activity is NOT observable from the OpenAI usage block or /v1/stats.
# Proving that speculative drafts were actually proposed/verified for a
# stochastic profile needs a separate, untimed runtime-log pass. A run whose
# draft path was suppressed must never be labelled as active dSpark.
DSPARK_EVIDENCE = {
    'usage_block_counters': False,
    'stats_endpoint_counters': False,
    'requires_runtime_log_debug_pass': True,
    'debug_pass_env': "RUST_LOG='info,ds41rt::timing=debug,ds41rt::cost_model=debug'",
    'debug_pass_perturbs_timing': True,
    'debug_pass_is_performance_evidence': False,
    'adaptive_gate_may_suppress_drafts': True,
    'record_actual_activity_do_not_force': True,
    'active_dspark_claim_requires': (
        'runtime-log evidence of draft proposal/verification for a stochastic '
        'profile; a suppressed-draft run must not be reported as active dSpark'),
}

# The raw measurement identity and the eventual report identity are distinct.
PROVENANCE_NOTE = (
    'source_revision is the measured source (git HEAD when this raw run was '
    'recorded); release_doc_commit is filled in only when a report is rendered '
    'and must never be conflated with the measurement source revision'
)


def token_zero_nonces(count, seed, tokenizer):
    start, total = seed % (0x9FFF - 0x4E00 + 1), 0x9FFF - 0x4E00 + 1
    result, seen = [], set()
    for offset in range(total):
        marker = chr(0x4E00 + ((start + offset) % total))
        prefix = f"{marker} request nonce {seed}-{len(result)}.\n"
        encoded = tokenizer.encode(prefix, add_special_tokens=False).ids
        marker_ids = tokenizer.encode(marker, add_special_tokens=False).ids
        if not encoded or len(marker_ids) != 1 or encoded[0] != marker_ids[0] or encoded[0] in seen:
            continue
        seen.add(encoded[0])
        result.append(dict(prefix=prefix, marker=marker, first_content_token_id=encoded[0]))
        if len(result) == count:
            return result
    raise RuntimeError(f"tokenizer exposed only {len(result)} unique token-zero nonces; need {count}")


def resolve_profile(name, overrides):
    """Canonical sampling vector for a named profile plus explicit overrides.

    Overrides are the individual `--temperature/--top-p/--top-k/--min-p` flags.
    Returns `(vector, source_name)` where `source_name` records whether the run
    used the profile verbatim or a documented override of it.
    """
    if name not in SAMPLING_PROFILES:
        raise SystemExit(f"unknown sampling profile {name!r}; "
                         f"choose from {', '.join(sorted(SAMPLING_PROFILES))}")
    vector = dict(SAMPLING_PROFILES[name])
    changed = []
    for field, value in overrides.items():
        if value is None:
            continue
        if field == 'top_k':
            # vLLM accepts -1 as a deprecated spelling of "disabled"; normalize
            # to the preferred 0 and keep the requested spelling visible.
            if value < -1:
                raise SystemExit("top_k must be 0 (disabled), -1 (disabled), or at least 1")
            if value in (-1, 0):
                value = 0
        vector[field] = value
        changed.append(field)
    if not math.isfinite(vector['temperature']) or vector['temperature'] < 0.0 or vector['temperature'] > 2.0:
        raise SystemExit("temperature must be finite and in [0, 2]")
    if not math.isfinite(vector['top_p']) or not 0.0 < vector['top_p'] <= 1.0:
        raise SystemExit("top_p must be finite and in (0, 1]")
    if vector['top_k'] != 0 and vector['top_k'] < 1:
        raise SystemExit("top_k must be 0 (disabled) or at least 1")
    if not math.isfinite(vector['min_p']) or not 0.0 <= vector['min_p'] <= 1.0:
        raise SystemExit("min_p must be finite and in [0, 1]")
    source = name if not changed else f"custom:{name}"
    return vector, source


def sampling_request_fields(vector):
    """Serialise the canonical sampling vector for the wire (strict only).

    Greedy follows the engine contract (`temperature < 1e-5` or `top_k == 1`).
    When `temperature < 1e-5`, temperature already selects greedy, so the body is
    temperature-only (the canonical control keeps the historical integer `0`) and
    every filter is inert. Otherwise, when `top_k == 1` is the trigger, the body
    is the full vector including `top_k == 1`; that field must be transmitted,
    because without it the request would no longer be greedy. Every stochastic
    request sends all four fields.
    """
    temperature = vector['temperature']
    if temperature < GREEDY_TEMPERATURE_EPSILON:
        return {'temperature': 0 if temperature == 0.0 else temperature}
    if vector['top_k'] == GREEDY_TOP_K:
        return {'temperature': temperature, 'top_p': vector['top_p'],
                'top_k': GREEDY_TOP_K, 'min_p': vector['min_p']}
    return {
        'temperature': temperature,
        'top_p': vector['top_p'],
        'top_k': vector['top_k'],
        'min_p': vector['min_p'],
    }


# The engine's greedy contract: temperature below this epsilon, or top_k == 1.
GREEDY_TEMPERATURE_EPSILON = 1e-5
GREEDY_TOP_K = 1


def is_greedy_vector(vector):
    """True when the engine would take its greedy path for this vector."""
    return (vector['temperature'] < GREEDY_TEMPERATURE_EPSILON
            or vector['top_k'] == GREEDY_TOP_K)


def greedy_effective_vector():
    """The sampling vector the engine applies on its greedy path."""
    return {'temperature': 0.0, 'top_p': 1.0, 'top_k': GREEDY_TOP_K, 'min_p': 0.0}


def effective_sampling_vector(vector):
    """The vector the engine actually applies, plus fields it does not apply.

    Greedy collapses to the engine's greedy constants regardless of the
    requested filters, so a reader can never mistake "requested" for "applied".
    `non_applied_fields` lists every requested field whose value differs from the
    applied one (for the canonical greedy control that is `top_k: 0` requested
    versus `top_k: 1` applied); it makes no claim about overrides.
    """
    effective = greedy_effective_vector() if is_greedy_vector(vector) else dict(vector)
    non_applied = [field for field in ('temperature', 'top_p', 'top_k', 'min_p')
                   if vector[field] != effective[field]]
    return effective, non_applied


def repeat_plan(repeats, repeat_index):
    """1-based repeat indices to run, validating an explicit repeat index."""
    if repeats < 1:
        raise SystemExit('repeats must be positive')
    if repeat_index is None:
        return list(range(1, repeats + 1))
    if repeat_index < 1 or repeat_index > repeats:
        raise SystemExit(f'repeat-index must be in 1..{repeats}')
    return [repeat_index]


def case_decode_limits(fixed_decode_tokens, weight, min_tokens, ignore_eos):
    """Per-case decode limits: (fixed_tokens, min_tokens, ignore_eos).

    `--fixed-decode-tokens` pins only WEIGHTED cases. Unweighted diagnostics
    (counting, orchid) keep their corpus budget and natural EOS, so the counting
    quality contract is not truncated. An explicit `--min-tokens`/`--ignore-eos`
    still applies to every case.
    """
    fixed = fixed_decode_tokens if weight > 0 else None
    return (fixed,
            fixed if fixed is not None else min_tokens,
            ignore_eos or fixed is not None)


def profile_publishable(samples):
    """Weighted publishability: non-empty weighted samples, every one passed.

    Unweighted diagnostics (counting, orchid) run at their corpus budget by
    design, so a diagnostic failure must not invalidate a weighted sampling
    comparison; it is recorded separately by `diagnostics_passed`. Comparison is
    fail-closed: only a literal True counts as a pass.
    """
    weighted = [sample for sample in samples if sample.get('weight', 0) > 0]
    return bool(weighted) and all(sample.get('passed') is True for sample in weighted)


def diagnostics_passed(samples):
    """True when every unweighted diagnostic sample passed (True if there are none)."""
    return all(sample.get('passed') is True
               for sample in samples if sample.get('weight', 0) <= 0)


def all_samples_passed(samples):
    """True only when the battery is non-empty and every sample (any weight) passed."""
    return bool(samples) and all(sample.get('passed') is True for sample in samples)


def effective_decode_limits(corpus, selected, fixed_decode_tokens, min_tokens, ignore_eos,
                            include_counting, include_orchid):
    """Effective `(min_tokens, max_tokens)` per selected case.

    Built through `case_decode_limits`, so a weighted case with a fixed length
    reports the fixed value for BOTH min and max (the fixed length overrides an
    explicit `--min-tokens`), while an unweighted diagnostic keeps its corpus
    max_tokens.
    """
    limits = {}
    case_ids = (list(selected)
                + (['counting'] if include_counting else [])
                + (['orchid'] if include_orchid else []))
    for case_id in case_ids:
        if case_id == 'counting':
            definition = corpus['counting']
        elif case_id == 'orchid':
            definition = corpus['orchid']
        else:
            definition = corpus['cases'][case_id]
        weight = 0.0 if case_id in ('counting', 'orchid') else definition['weight']
        case_fixed, case_min, _ = case_decode_limits(
            fixed_decode_tokens, weight, min_tokens, ignore_eos)
        limits[case_id] = (case_min,
                           case_fixed if case_fixed is not None else definition['max_tokens'])
    return limits


def over_budget_cases(limits):
    """Cases whose effective min_tokens would exceed their effective max_tokens."""
    return {case: pair for case, pair in limits.items()
            if pair[0] is not None and pair[0] > pair[1]}


def weighted_decode_metric(samples):
    """Weight-normalized decode-only throughput for a set of timed samples.

    Mirrors the release corpus contract exactly:

      sum(weight * (completion_tokens - 1)) / sum(weight * post_first_token_s)

    `completion_tokens - 1` drops the first token, which is charged to prefill
    rather than decode; `post_first_token_seconds` is `finish_seconds -
    first_output_seconds`, so reasoning tokens are never charged only to
    final-answer time. Samples without timing, or with weight 0 (counting and
    the historical orchid diagnostic), are excluded.
    """
    weighted = [s for s in samples
                if s.get('weight', 0) > 0
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


def harness_identity(script_path):
    """Identity of the harness that produced a raw report."""
    path = Path(script_path).resolve()
    return {'path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def source_revision(repo_root):
    """Best-effort git HEAD of the measured source; None when unavailable."""
    try:
        result = subprocess.run(['git', '-C', str(repo_root), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=10, check=True)
    except Exception:
        return None
    revision = result.stdout.strip()
    return revision or None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-V4.1-Flash')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--corpus', type=Path,
                        default=Path(__file__).with_name('fixtures') / 'release-semantic-corpus.json')
    parser.add_argument('--label', required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--repeat-index', type=int,
                        help='Run exactly this 1-based repeat index instead of repeats '
                             '1..--repeats. Used by the interleaved profile rotation so a '
                             'deployment drift hits every profile roughly equally; the '
                             'per-repeat nonce identity is preserved regardless of which '
                             'invocation runs a repeat.')
    parser.add_argument('--nonce-seed', type=int, required=True)
    parser.add_argument('--case', action='append')
    parser.add_argument('--include-orchid', action='store_true', help='Historical diagnostic only')
    parser.add_argument('--include-counting', action='store_true')
    parser.add_argument('--counting-only', action='store_true')
    parser.add_argument('--api-key-env', help='Environment variable containing the API key; never saved')
    parser.add_argument('--remote-reference', action='store_true', help='Record provider cache counters without enforcing local cache policy')
    parser.add_argument('--output', type=Path, required=True)

    identity = parser.add_argument_group('run identity')
    identity.add_argument('--identity-file', type=Path,
                          help='JSON snapshot of hardware, software and container-image '
                               'identity captured before the run. Embedded under '
                               'provenance.identity with its SHA-256. A release campaign '
                               'requires it: git HEAD alone is not identity.')

    sampling = parser.add_argument_group('sampling comparison')
    sampling.add_argument('--sampling-profile', choices=sorted(SAMPLING_PROFILES),
                          default=DEFAULT_SAMPLING_PROFILE,
                          help='Named canonical sampling vector (default: greedy, the historical control)')
    sampling.add_argument('--seed', type=int,
                          help='Per-request sampling seed (required for stochastic profiles)')
    sampling.add_argument('--unseeded', action='store_true',
                          help='Explicitly allow a stochastic profile without a seed; the '
                               'report is marked non-reproducible')
    sampling.add_argument('--temperature', type=float)
    sampling.add_argument('--top-p', type=float)
    sampling.add_argument('--top-k', type=int)
    sampling.add_argument('--min-p', type=float)
    sampling.add_argument('--list-profiles', action='store_true',
                          help='Print the canonical profile vectors as JSON and exit')

    decode = parser.add_argument_group('decode-length policy')
    decode.add_argument('--ignore-eos', action='store_true',
                        help='Request ignore_eos: continue to the output budget instead of '
                             'stopping at a model stop token')
    decode.add_argument('--min-tokens', type=int,
                        help='Request min_tokens: ignore stop tokens until this many output '
                             'tokens are committed')
    decode.add_argument('--fixed-decode-tokens', type=int,
                        help='Pin every sample to exactly N generated tokens by setting '
                             'ignore_eos, min_tokens=N and max_tokens=N')
    # `--list-profiles` is a documentation surface and must work without a
    # server, tokenizer or the otherwise-required request arguments.
    if '--list-profiles' in sys.argv[1:]:
        print(json.dumps(SAMPLING_PROFILES, indent=2, sort_keys=True))
        return
    args = parser.parse_args()

    if args.output.exists():
        parser.error('output already exists')
    if args.fixed_decode_tokens is not None and args.fixed_decode_tokens < 1:
        parser.error('fixed-decode-tokens must be positive')
    if args.min_tokens is not None and args.min_tokens < 0:
        parser.error('min-tokens must be non-negative')
    try:
        repeat_indices = repeat_plan(args.repeats, args.repeat_index)
    except SystemExit as error:
        parser.error(str(error))

    identity_snapshot = None
    identity_sha256 = None
    if args.identity_file is not None:
        if not args.identity_file.is_file():
            parser.error(f'identity file not found: {args.identity_file}')
        raw_identity = args.identity_file.read_bytes()
        try:
            identity_snapshot = json.loads(raw_identity)
        except json.JSONDecodeError as error:
            parser.error(f'identity file is not valid JSON: {error}')
        identity_sha256 = hashlib.sha256(raw_identity).hexdigest()

    vector, vector_source = resolve_profile(
        args.sampling_profile,
        dict(temperature=args.temperature, top_p=args.top_p,
             top_k=args.top_k, min_p=args.min_p))
    sampling_fields = sampling_request_fields(vector)
    effective_vector, non_applied_fields = effective_sampling_vector(vector)

    greedy = is_greedy_vector(vector)
    if not greedy and args.seed is None and not args.unseeded:
        parser.error('a stochastic sampling profile requires --seed for reproducibility '
                     '(or --unseeded to record a non-reproducible run)')
    seed = None if greedy else args.seed

    fixed_decode_tokens = args.fixed_decode_tokens

    corpus = json.loads(args.corpus.read_text())
    selected = [] if args.counting_only else (args.case or corpus['weighted_case_ids'])
    if args.counting_only:
        args.include_counting = True
    unknown = set(selected) - set(corpus['weighted_case_ids'])
    if unknown:
        parser.error(f"unknown or non-weighted cases: {sorted(unknown)}")
    # Fail early rather than emit min_tokens > max_tokens, which the server would
    # reject. Limits are computed through case_decode_limits, so a fixed length
    # that overrides --min-tokens for weighted cases is not a false positive.
    limits = effective_decode_limits(corpus, selected, fixed_decode_tokens,
                                     args.min_tokens, args.ignore_eos,
                                     args.include_counting, args.include_orchid)
    over_budget = over_budget_cases(limits)
    if over_budget:
        parser.error(f"--min-tokens exceeds the effective output budget for "
                     f"{over_budget}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    # The nonce for (repeat, case) is a pure function of its position in the
    # full repeat plan, so running one repeat in isolation (interleaved profile
    # rotation) yields the same prompt identity as the in-process run.
    cases_per_repeat = len(selected) + int(args.include_orchid) + int(args.include_counting)
    nonces = token_zero_nonces(args.repeats * cases_per_repeat, args.nonce_seed, tokenizer)
    quality = runpy.run_path(str(Path(__file__).with_name('release_throughput_checks.py')))
    api = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))

    sampling_metadata = dict(
        profile=args.sampling_profile,
        profile_source=vector_source,
        requested_vector=vector,
        effective_vector=effective_vector,
        non_applied_fields=non_applied_fields,
        request_fields=sampling_fields,
        encoding='strict',
        seed=seed,
        seed_source=('explicit' if seed is not None else
                     ('greedy-ignores-seed' if greedy else 'unseeded-non-reproducible')),
        greedy=greedy,
        semantics=SAMPLING_SEMANTICS,
        server_defaults_if_unset=SERVER_DEFAULTS_IF_UNSET,
        # The native API does not echo the sampling parameters it applied, so
        # "accepted" cannot be promoted to "honoured" from the report alone.
        applied_sampling_echo=None,
    )
    decode_policy = dict(
        fixed_decode_tokens=fixed_decode_tokens,
        fixed_applies_to='weighted cases only; counting/orchid keep their corpus budgets',
        cli_min_tokens=args.min_tokens,
        cli_ignore_eos=args.ignore_eos,
        fixed_overrides_min_tokens=bool(
            fixed_decode_tokens is not None and args.min_tokens is not None),
        note=('fixed length requested for weighted cases; honoured only if those samples '
              'report exactly requested_decode_tokens completion tokens and a length finish. '
              'Unweighted diagnostics (counting, orchid) keep their corpus max_tokens and '
              'natural EOS so the counting quality contract is not truncated, and their '
              'actual token counts are recorded. --fixed-decode-tokens overrides an explicit '
              '--min-tokens for the fixed cases.'
              if fixed_decode_tokens is not None else
              'no fixed length requested; stochastic output lengths are recorded per sample'),
    )
    report = dict(
        scope=__doc__, label=args.label, base_url=args.base_url, model=args.model,
        corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        repeats=args.repeats, nonce_seed=args.nonce_seed, selected_cases=selected,
        include_orchid=args.include_orchid, include_counting=args.include_counting,
        remote_reference=args.remote_reference,
        controls=dict(temperature=vector['temperature'], thinking='per-case; default disabled'),
        sampling=sampling_metadata,
        decode_length_policy=decode_policy,
        dspark_evidence=DSPARK_EVIDENCE,
        provenance=dict(
            source_label=args.label,
            source_revision=source_revision(Path(__file__).resolve().parents[1]),
            harness=harness_identity(__file__),
            identity=identity_snapshot,
            identity_sha256=identity_sha256,
            identity_required_for_release_campaign=True,
            release_doc_commit=None,
            note=PROVENANCE_NOTE,
        ),
        samples=[], repeat_summaries=[], case_summaries=[], passed=False,
        publishable=False,
    )

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        parser.error('API key environment variable is empty')
    for repeat in repeat_indices:
        repeat_samples = []
        nonce_index = (repeat - 1) * cases_per_repeat
        for case_id in selected + (['orchid'] if args.include_orchid else []) + (['counting'] if args.include_counting else []):
            nonce = nonces[nonce_index]
            nonce_index += 1
            if case_id == 'counting':
                definition = corpus['counting']
                prompt = definition['prompt']
                max_tokens, weight, response_format = definition['max_tokens'], 0.0, None
            elif case_id == 'orchid':
                definition = corpus['orchid']
                prompt = definition['prompt_template'].format(nonce=nonce['prefix'].strip())
                max_tokens, weight = definition['max_tokens'], 0.0
                response_format = None
            else:
                definition = corpus['cases'][case_id]
                prompt = nonce['prefix'] + definition['prompt']
                max_tokens, weight = definition['max_tokens'], definition['weight']
                response_format = None
                if definition['json_schema']:
                    response_format = dict(type='json_schema', json_schema=dict(
                        name='file_edit', strict=True, schema=corpus['structured_edit_schema']))
            # `--fixed-decode-tokens` pins only WEIGHTED cases. The unweighted
            # diagnostics (counting, orchid) keep their corpus budget and natural
            # EOS: pinning counting to 256 would truncate the count_to=200
            # sequence, and forcing ignore_eos would run it past completion — a
            # quality failure for a case that is outside the weighted score.
            case_fixed, case_min_tokens, case_ignore_eos = case_decode_limits(
                fixed_decode_tokens, weight, args.min_tokens, args.ignore_eos)
            if case_fixed is not None:
                max_tokens = case_fixed
            body = dict(model=args.model, messages=[dict(role='user', content=prompt)],
                        thinking=dict(type=definition.get('thinking', 'disabled')), max_tokens=max_tokens,
                        stream=True, stream_options=dict(include_usage=True))
            body.update(sampling_fields)
            if not greedy:
                # Sending an explicit seed keeps the run reproducible; a greedy
                # request does not consume it, so it is deliberately omitted.
                body['seed'] = seed
            if case_min_tokens is not None:
                body['min_tokens'] = case_min_tokens
            if case_ignore_eos:
                body['ignore_eos'] = True
            if definition.get('reasoning_effort'):
                body['reasoning_effort'] = definition['reasoning_effort']
            if response_format:
                body['response_format'] = response_format
            sample = dict(repeat=repeat, case=case_id, category=definition.get('category', 'low-entropy'),
                          weight=weight, nonce=None if case_id == 'counting' else nonce, request=body,
                          requested_max_tokens=max_tokens,
                          requested_min_tokens=case_min_tokens,
                          requested_ignore_eos=case_ignore_eos,
                          requested_decode_tokens=case_fixed,
                          started_ns=time.time_ns())
            report['samples'].append(sample)
            save()
            try:
                result = api['stream_case'](args.base_url, body, api_key=api_key)
                content = result.pop('text')
                events = result.pop('events')
                finish_event = next(event['event'] for event in reversed(events)
                                    if any(c.get('finish_reason') for c in event['event'].get('choices', [])))
                sample.update(result, content=content,
                              content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                              finish_reason=finish_event['choices'][0]['finish_reason'],
                              system_fingerprint=finish_event.get('system_fingerprint'),
                              cached_tokens=result['usage'].get('prompt_cache_hit_tokens',
                                  result['usage'].get('prompt_tokens_details', {}).get('cached_tokens')))
                if case_id == 'counting':
                    expected = [str(n) for n in range(1, definition['count_to'] + 1)]
                    sample['quality_contract_passed'] = [x.strip() for x in content.strip().split(',')] == expected
                    sample['quality_contract_issues'] = [] if sample['quality_contract_passed'] else ['incorrect counting sequence']
                elif case_id == 'orchid':
                    words = content.split()
                    sample['quality_contract_passed'] = words == ['orchid'] * definition['requested_repetitions']
                    sample['quality_contract_issues'] = ([] if sample['quality_contract_passed'] else
                        [f"expected {definition['requested_repetitions']} exact orchid words, observed {len(words)} tokens"])
                else:
                    sample.update(quality['check_output'](case_id, content))
                # The first user-content token is unique. A small hit may still
                # cover the invariant chat-template prefix before user content.
                sample['bounded_static_prefix_hit'] = (isinstance(sample['cached_tokens'], int) and 0 <= sample['cached_tokens'] <= 32)
                sample['reasoning_present'] = bool(sample.get('reasoning', '').strip())
                sample['serving_completed'] = bool(content.strip())
                sample['completion_tokens'] = sample['usage'].get('completion_tokens')
                sample['fixed_length_honored'] = (
                    None if sample['requested_decode_tokens'] is None
                    else (sample['completion_tokens'] == sample['requested_decode_tokens']
                          and sample['finish_reason'] == 'length'))
                sample['passed'] = (sample['serving_completed']
                    and (definition.get('thinking') != 'enabled' or sample['reasoning_present'])
                    and sample.get('quality_contract_passed', True)
                    and sample.get('objective_checks_passed') is not False
                    and (args.remote_reference or case_id == 'counting' or sample['bounded_static_prefix_hit']))
            except Exception as error:
                sample.update(error=repr(error), passed=False)
            save()
            repeat_samples.append(sample)
            actual = sample.get('completion_tokens')
            print(args.label, repeat, case_id, 'PASS' if sample['passed'] else 'FAIL',
                  f'completion_tokens={actual}', flush=True)
        metric = weighted_decode_metric(repeat_samples)
        weighted = [sample for sample in repeat_samples
                    if sample['weight'] > 0 and 'finish_seconds' in sample]
        weighted_cases = metric['weighted_cases']
        timed_rows = [sample for sample in repeat_samples if 'finish_seconds' in sample]
        completion_tokens = [sample['usage']['completion_tokens'] for sample in timed_rows]
        summary = dict(repeat=repeat, weighted_cases=weighted_cases,
                       timed_tokens=metric['timed_tokens'],
                       timed_seconds=metric['timed_seconds'],
                       serving_completed=sum(sample.get('serving_completed', False) for sample in weighted),
                       objective_checks_passed=sum(sample.get('objective_checks_passed') is True for sample in weighted),
                       objective_checks_assessed=sum(sample.get('objective_checks_passed') is not None for sample in weighted),
                       weighted_observed_decode_tokens_per_second=(
                           metric['weighted_observed_decode_tokens_per_second']
                           if weighted_cases == len(selected) and selected else None),
                       partial_weighted_observed_decode_tokens_per_second=metric['weighted_observed_decode_tokens_per_second'],
                       complete_weighted_corpus=weighted_cases == len(selected) and bool(selected),
                       all_samples_passed=all(sample['passed'] for sample in repeat_samples),
                       completion_tokens_min=min(completion_tokens) if completion_tokens else None,
                       completion_tokens_max=max(completion_tokens) if completion_tokens else None,
                       fixed_length_honored=sum(sample.get('fixed_length_honored') is True for sample in repeat_samples),
                       fixed_length_assessed=sum(sample.get('fixed_length_honored') is not None for sample in repeat_samples))
        report['repeat_summaries'].append(summary)
        save()
    values = [item['weighted_observed_decode_tokens_per_second'] for item in report['repeat_summaries']
              if item['weighted_observed_decode_tokens_per_second'] is not None]
    report['median_weighted_observed_decode_tokens_per_second'] = statistics.median(values) if values else None

    # Per-case medians across repeats, so a single slow repeat cannot hide
    # behind the weighted aggregate and stochastic length spread stays visible.
    case_summaries = []
    for case_id in selected + (['orchid'] if args.include_orchid else []) + (['counting'] if args.include_counting else []):
        rows = [row for row in report['samples']
                if row['case'] == case_id and 'observed_decode_tokens_per_second' in row
                and row['observed_decode_tokens_per_second'] is not None]
        if not rows:
            continue
        tps = [row['observed_decode_tokens_per_second'] for row in rows]
        tokens = [row['usage']['completion_tokens'] for row in rows]
        definition = corpus['counting'] if case_id == 'counting' else (
            corpus['orchid'] if case_id == 'orchid' else corpus['cases'][case_id])
        case_summaries.append(dict(
            case=case_id,
            category=definition.get('category', 'low-entropy'),
            weight=0.0 if case_id in ('counting', 'orchid') else definition['weight'],
            samples=len(rows),
            median_observed_decode_tokens_per_second=statistics.median(tps),
            min_observed_decode_tokens_per_second=min(tps),
            max_observed_decode_tokens_per_second=max(tps),
            median_completion_tokens=statistics.median(tokens),
            min_completion_tokens=min(tokens),
            max_completion_tokens=max(tokens),
            samples_passed=sum(row['passed'] for row in rows),
            fixed_length_honored=sum(row.get('fixed_length_honored') is True for row in rows),
        ))
    report['case_summaries'] = case_summaries
    report['weighted_passed'] = profile_publishable(report['samples'])
    report['diagnostics_passed'] = diagnostics_passed(report['samples'])
    report['all_samples_passed'] = all_samples_passed(report['samples'])
    # A profile may be published ONLY when its weighted comparison passed. A
    # failing quality check stays in the weighted rate on purpose (the release
    # protocol keeps failures visible rather than discarding them), so
    # `publishable` is the explicit gate for any downstream report or README
    # table. Diagnostics are recorded but never gate it, and a failed weighted
    # profile exits nonzero as well.
    report['quality_failures_included_in_weighted_metric'] = True
    report['publishability_basis'] = ('weighted cases only (weight > 0); diagnostics are '
                                      'recorded in diagnostics_passed and do not gate publishable')
    report['publishable'] = report['weighted_passed']
    report['passed'] = report['weighted_passed']
    save()
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
