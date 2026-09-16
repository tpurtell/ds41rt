import runpy
import pytest
from pathlib import Path

MODULE = runpy.run_path(str(Path(__file__).parents[1] / 'summarize-ds41-native-policy.py'))


def test_conditional_acceptance_censors_rejected_history_and_terminal_rows():
    base = dict(request_id=1, lane=0, generated=1, rows=6, confidence=[0.] * 5,
                constrained=False, terminal=False)
    rows = [dict(base, matched=1), dict(base, matched=5),
            dict(base, matched=0, terminal=True), dict(base, matched=0, constrained=True)]
    result = MODULE['summarize'](rows, [])['calibration']
    assert [r['samples'] for r in result] == [2, 2, 1, 1, 1]
    assert result[0]['bins'][0]['observed_acceptance'] == 1
    assert result[1]['bins'][0]['observed_acceptance'] == .5
    assert result[2]['bins'][0]['observed_acceptance'] == 1


def test_short_full_match_does_not_label_unverified_suffix():
    row = dict(request_id=1, lane=1, generated=1, rows=2, confidence=[0.] * 5,
               constrained=False, terminal=False, matched=1)
    result = MODULE['summarize']([row], [])['calibration']
    assert [r['samples'] for r in result] == [1, 0, 0, 0, 0]


def trace(rows=8, matched=7):
    return (f'native draft policy observation request_id=1 lane=0 generated=10 '
            f'verifier_rows={rows} matched_prefix={matched} raw_confidence=[0,0,0,0,0,0,0] '
            'constrained=false eos=false length_limit=false')


def test_k7_summary_includes_last_two_positions_and_explicit_denominators():
    observations, rounds = MODULE['parse'](trace() + '\n' + trace(matched=5))
    result = MODULE['summarize'](observations, rounds)
    assert [r['samples'] for r in result['calibration']] == [2, 2, 2, 2, 2, 2, 1]
    acceptance = result['acceptance']
    assert acceptance['proposed_drafts'] == 14
    assert acceptance['accepted_drafts'] == 12
    assert acceptance['accepted_fraction'] == 12 / 14
    assert acceptance['verified_width_counts'] == {7: 2}


@pytest.mark.parametrize('rows,matched', [(9, 7), (8, 8), (0, 0), (8, -1)])
def test_parser_rejects_impossible_observations(rows, matched):
    with pytest.raises(ValueError, match='invalid verified prefix'):
        MODULE['parse'](trace(rows, matched))


def test_terminal_observations_do_not_lower_acceptance():
    observations, _ = MODULE['parse'](trace() + '\n' + trace(matched=0).replace('eos=false', 'eos=true'))
    result = MODULE['summarize'](observations, [])['acceptance']
    assert result['accepted_fraction'] == 1
    assert result['observations'] == 1
    assert result['excluded_terminal_or_constrained'] == 1


def test_independent_lane_round_keeps_existing_cost_fields():
    _, rounds = MODULE['parse']('native independent lane round lane=1 requests=8 '
        'proposed=56 accepted=24 emitted=24 draft_us=6270 prepared_us=6325 '
        'verify_us=73107 total_us=83340')
    assert rounds == [dict(requests=8, proposed=56, accepted=24, draft_us=6270,
                           prepare_us=6325, verify_us=73107, total_us=83340)]


def test_zero_acceptance_and_emissions_use_only_nonterminal_observations():
    observations, _ = MODULE['parse'](trace(matched=0) + '\n' + trace(matched=4)
        + '\n' + trace(matched=0).replace('length_limit=false', 'length_limit=true'))
    result = MODULE['summarize'](observations, [])['acceptance']
    assert result['zero_acceptance_observations'] == 1
    assert result['zero_acceptance_fraction'] == .5
    assert result['mean_emitted_tokens'] == 3
    assert result['mean_accepted_drafts'] == 2


def test_no_eligible_observations_reports_null_rates():
    observations, _ = MODULE['parse'](trace().replace('eos=false', 'eos=true'))
    result = MODULE['summarize'](observations, [])['acceptance']
    assert result['zero_acceptance_observations'] == 0
    assert result['zero_acceptance_fraction'] is None
    assert result['mean_emitted_tokens'] is None
