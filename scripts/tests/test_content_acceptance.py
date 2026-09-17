"""Acceptance denominators must not mix grammar targets or censored endings."""
import runpy
from pathlib import Path

summarize = runpy.run_path(str(Path(__file__).parents[1] /
                             'collect-ds41-content-acceptance.py'))['summarize_acceptance']


def test_separates_grammar_and_terminal_cycles():
    base = dict(request_id=1, rows=4, matched=2, terminal=False, constrained=False)
    result = summarize([base, dict(base, matched=0),
                        dict(base, matched=0, terminal=True),
                        dict(base, request_id=2, matched=3, constrained=True)])
    plain = result['unconstrained']
    assert plain['verified_drafts'] == 6
    assert plain['accepted_drafts'] == 2
    assert plain['acceptance'] == 1 / 3
    assert plain['mean_emitted_tokens'] == 2
    assert plain['excluded_terminal_observations'] == 1
    assert plain['zero_acceptance_cycles'] == 1
    assert result['grammar_constrained']['acceptance'] == 1


def test_adaptive_zero_draft_round_is_not_a_failed_draft():
    row = dict(request_id=1, rows=1, matched=0, terminal=False, constrained=False)
    plain = summarize([row])['unconstrained']
    assert plain['acceptance'] is None
    assert plain['mean_emitted_tokens'] == 1
    assert plain['verified_width_counts'] == {'0': 1}
    assert summarize([])['grammar_constrained']['mean_emitted_tokens'] is None
