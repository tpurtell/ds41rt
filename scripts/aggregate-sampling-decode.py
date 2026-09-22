#!/usr/bin/env python3
"""Aggregate interleaved sampling-comparison decode reports (CPU only).

The campaign runs the five sampling profiles interleaved by repeat so a slow
drift in the deployment hits every profile roughly equally::

  repeat 1: greedy -> temp0.2-topp0.95 -> temp0.7-topp0.9 -> temp0.7-minp0.05 -> temp0.7-topk40
  repeat 2: temp0.7-topk40 -> greedy -> temp0.2-topp0.95 -> ...   (rotated)
  repeat 3: temp0.7-minp0.05 -> ...                              (rotated)

Each invocation of ``scripts/bench-ds41-release-decode.py`` writes one raw
report (use ``--repeat-index`` to run a single repeat). This tool combines those
raw reports into per-profile weighted medians, per-case medians and the
*observed* execution order, so a profile-major ordering (all repeats of greedy,
then all repeats of the next profile) is visible rather than silently averaged.

Nothing here contacts a server or a GPU, and no number is invented: every value
is read from a raw report whose SHA-256 is recorded in the output.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def load_report(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def report_start_ns(document):
    """Earliest recorded sample start, used only to order the raw runs."""
    values = [s.get('started_ns') for s in document.get('samples', [])
              if isinstance(s.get('started_ns'), int)]
    return min(values) if values else None


def repeat_start_times(document):
    """Earliest sample start per repeat, so a multi-repeat report keeps order.

    A single document-level minimum would collapse every repeat in that document
    to one timestamp, making the observed execution order alphabetical by
    profile. The harness tags each sample with its 1-based `repeat`, so order by
    that instead. Falls back to the document minimum when samples carry no
    repeat tag.
    """
    times = {}
    for sample in document.get('samples', []):
        repeat = sample.get('repeat')
        started = sample.get('started_ns')
        if not isinstance(started, int) or repeat is None:
            continue
        if repeat not in times or started < times[repeat]:
            times[repeat] = started
    return times


def profile_sequence_is_major(sequence):
    """True when every profile's runs are contiguous in time (not interleaved)."""
    for profile in set(sequence):
        positions = [index for index, value in enumerate(sequence) if value == profile]
        if positions and max(positions) - min(positions) + 1 != len(positions):
            return False
    return True


def aggregate(documents):
    """Combine ``(document, sha256, path)`` triples into one report."""
    entries = []
    samples_by_profile = {}
    for document, sha256, path in documents:
        profile = (document.get('sampling') or {}).get('profile')
        if not profile:
            raise ValueError(f"{path}: not a sampling-comparison decode report")
        start_times = repeat_start_times(document)
        fallback_start = report_start_ns(document)
        for summary in document.get('repeat_summaries', []):
            entries.append(dict(
                profile=profile,
                repeat=summary.get('repeat'),
                weighted_tps=summary.get('weighted_observed_decode_tokens_per_second'),
                partial_weighted_tps=summary.get('partial_weighted_observed_decode_tokens_per_second'),
                timed_tokens=summary.get('timed_tokens'),
                timed_seconds=summary.get('timed_seconds'),
                source=str(path),
                source_sha256=sha256,
                start_ns=start_times.get(summary.get('repeat'), fallback_start),
            ))
        samples_by_profile.setdefault(profile, []).extend(document.get('samples', []))

    profiles = {}
    for profile in sorted({entry['profile'] for entry in entries}):
        ordered = sorted((entry for entry in entries if entry['profile'] == profile),
                         key=lambda entry: (entry['repeat'] is None, entry['repeat']))
        values = [entry['weighted_tps'] for entry in ordered if entry['weighted_tps'] is not None]
        token_totals = [entry['timed_tokens'] for entry in ordered if entry['timed_tokens'] is not None]
        second_totals = [entry['timed_seconds'] for entry in ordered if entry['timed_seconds'] is not None]
        profiles[profile] = {
            'repeats': [entry['repeat'] for entry in ordered],
            'weighted_observed_decode_tokens_per_second': values,
            'median_weighted_observed_decode_tokens_per_second':
                statistics.median(values) if values else None,
            'min_weighted_observed_decode_tokens_per_second': min(values) if values else None,
            'max_weighted_observed_decode_tokens_per_second': max(values) if values else None,
            'timed_tokens_total': sum(token_totals) if token_totals else None,
            'timed_seconds_total': sum(second_totals) if second_totals else None,
            'sources': [entry['source'] for entry in ordered],
        }

    case_medians = {}
    for profile, rows in sorted(samples_by_profile.items()):
        by_case = {}
        for row in rows:
            if row.get('observed_decode_tokens_per_second') is None:
                continue
            by_case.setdefault(row.get('case'), []).append(row)
        case_medians[profile] = {
            case: {
                'samples': len(case_rows),
                'median_observed_decode_tokens_per_second':
                    statistics.median([r['observed_decode_tokens_per_second'] for r in case_rows]),
                'median_completion_tokens':
                    statistics.median([r['usage']['completion_tokens'] for r in case_rows]),
                'samples_passed': sum(bool(r.get('passed')) for r in case_rows),
            }
            for case, case_rows in sorted(by_case.items(), key=lambda item: str(item[0]))
        }

    timed = sorted((entry for entry in entries if entry['start_ns'] is not None),
                   key=lambda entry: (entry['start_ns'], entry['profile'], entry['repeat'] or 0))
    sequence = [entry['profile'] for entry in timed]
    return {
        'schema': 1,
        'scope': __doc__,
        'profiles': profiles,
        'weighted_ranking': sorted(
            (profile for profile in profiles
             if profiles[profile]['median_weighted_observed_decode_tokens_per_second'] is not None),
            key=lambda profile: profiles[profile]['median_weighted_observed_decode_tokens_per_second'],
            reverse=True,
        ),
        'case_medians': case_medians,
        'execution_order': [
            {'order': index + 1, 'profile': entry['profile'], 'repeat': entry['repeat'],
             'start_ns': entry['start_ns'], 'source': entry['source']}
            for index, entry in enumerate(timed)
        ],
        'profile_major': profile_sequence_is_major(sequence) if sequence else None,
        'interleaved': (not profile_sequence_is_major(sequence)) if sequence else None,
        'complete_profiles': sorted(profiles),
        'inputs': [{'path': str(path), 'sha256': sha256} for _, sha256, path in documents],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True,
                        help='Raw reports from bench-ds41-release-decode.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='sampling-comparison')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    documents = []
    for path in args.inputs:
        if not path.is_file():
            parser.error(f'input report not found: {path}')
        document, sha256 = load_report(path)
        documents.append((document, sha256, path))
    result = aggregate(documents)
    result['label'] = args.label
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({profile: result['profiles'][profile]
                      ['median_weighted_observed_decode_tokens_per_second']
                      for profile in result['complete_profiles']}, indent=2))
    print(f"interleaved={result['interleaved']} profiles={len(result['complete_profiles'])}")


if __name__ == '__main__':
    main()
