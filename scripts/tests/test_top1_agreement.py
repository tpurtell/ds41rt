"""Protocol checks for the one-time v5 teacher-forced top-1 study."""
import hashlib
import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'scripts/collect-ds41-top1-agreement.py'
CORPUS = ROOT/'scripts/fixtures/release-v5-top1-corpus.json'
COLLECT = runpy.run_path(str(SCRIPT))


def test_corpus_is_frozen_diverse_and_teacher_forced():
    raw = CORPUS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '2c30b0c840160ac02a859e177fc0d7e0dad3e6dcab1d67c57bc5b2d6d26c6e45'
    corpus = json.loads(raw)
    samples = COLLECT['validate_corpus'](corpus)
    assert len(samples) == 132
    assert set(corpus['category_counts'].values()) == {12}
    assert len(corpus['category_counts']) == 11
    assert len({json.dumps(row['messages'], sort_keys=True, ensure_ascii=False) for row in samples}) == 132
    assert min(len(row['messages'][-1]['content']) for row in samples) > 20
    assert min(len(row['messages'][0]['content']) for row in samples
               if row['category'] == 'long-context') > 5_000


def test_trace_parser_uses_exact_token_ids_and_rejects_missing_fields():
    trace = (b'2026-09-17 INFO ds41rt::top1_agreement: native prompt top-1 '
             b'request_id=41 prompt_tokens=309 token_id=1287\n')
    assert COLLECT['parse_top1'](trace) == [dict(request_id=41, prompt_tokens=309,
        token_id=1287, line=trace.decode().strip())]
    with pytest.raises(KeyError):
        COLLECT['parse_top1'](b'native prompt top-1 request_id=1 prompt_tokens=2\n')


def test_agreement_summary_reports_denominator_and_wilson_interval():
    rows = [dict(matches_reference=value, reference_token_id=i%3, token_id=i%4)
            for i, value in enumerate([True]*9 + [False])]
    result = COLLECT['summary'](rows)
    assert result['samples'] == 10 and result['matches'] == 9 and result['agreement'] == .9
    assert result['baseline_token_count'] == 3 and result['candidate_token_count'] == 4
    lower, upper = result['wilson_95']
    assert lower < .9 < upper


def test_reference_requires_same_frozen_corpus(tmp_path):
    corpus = json.loads(CORPUS.read_text())
    ids = [row['id'] for row in corpus['samples']]
    reference = tmp_path/'baseline.json'
    reference.write_text(json.dumps(dict(passed=True, role='baseline', corpus_sha256='bad',
        protocol={'target_only':True}, samples=[{'id':key,'token_id':1} for key in ids])))
    with pytest.raises(AssertionError):
        COLLECT['load_reference'](reference, 'expected', ids)
