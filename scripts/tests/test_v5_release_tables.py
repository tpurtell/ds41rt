"""Synthetic layout checks only; historical numbers are never release inputs."""
import copy
import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[2]
RENDER = runpy.run_path(str(ROOT/'scripts/render-ds41-v5-release-tables.py'))


def reports():
    old = json.loads((ROOT/'docs/release-v4-performance.json').read_text())
    result = {}
    for model in ('full', 'exl3'):
        value = copy.deepcopy(old)
        value['model'] = model
        value['weighted_case_ids'] = list(set(RENDER['CASES']) - {'counting'})
        value['controls']['case_controls'] = {'code-reasoning': {'thinking': 'enabled', 'reasoning_effort': 'high'}}
        for layout in ('single', 'dual'):
            for mode in ('decode', 'target_decode'):
                value['layouts'][layout][mode]['cases']['code-reasoning'] = dict(samples=3, median_tps=123., min_tps=122., max_tps=124.)
        result[model] = value
    return result


def test_four_configuration_tables_without_old_comparisons_or_peaks():
    data = reports()
    tools = json.loads((ROOT/'docs/sparkinfer-upstream-tool-eval-20260916.json').read_text())
    # Current collectors write summaries.json as a raw three-run list; older
    # preserved evidence wraps the same rows in a provenance object.
    text = RENDER['render'](data, tools={'full': tools['runs'], 'exl3': tools,
                                        'exl3-fp4ple': tools['runs']})
    assert text.count(' prefill matrix.**') == 4
    assert text.count(' content-type decode.**') == 2
    assert text.count(' concurrency scaling.**') == 2
    assert text.count('| Code with reasoning |') == 2
    memory = text.split('**Memory after readiness.**')[1].split('**Tool calling.**')[0]
    assert sum(line.startswith(('| Full ', '| EXL3 ')) for line in memory.splitlines()) == 6
    tools_section = text.split('**Tool calling.**')[1]
    assert sum(line.startswith(('| Full /', '| EXL3 /')) for line in tools_section.splitlines()) == 9
    assert 'Change from v3' not in text and 'short-prefill controls' not in text and 'Observed GPU peaks' not in text
    assert '400 W' in text and 'standard memory speed' in text and 'private-tail tokens' in text


@pytest.mark.parametrize('defect', ['missing_reasoning', 'corpus_mismatch', 'binary_mismatch', 'duplicate_concurrency'])
def test_incomplete_or_unmatched_measurements_are_rejected(defect):
    data = reports()
    if defect == 'missing_reasoning':
        del data['exl3']['layouts']['dual']['decode']['cases']['code-reasoning']
    elif defect == 'corpus_mismatch':
        data['exl3']['corpus_sha256'] = 'different'
    elif defect == 'binary_mismatch':
        data['exl3']['binary_sha256'] = 'different'
    else:
        rows = data['exl3']['layouts']['single']['concurrency']['code']['summaries']
        rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(AssertionError):
        RENDER['render'](data)


def test_acceptance_keeps_grammar_targets_separate_and_rejects_invalid_denominator():
    data = reports()
    rate = dict(nonterminal_observations=10, verified_drafts=20, accepted_drafts=10,
                acceptance=.5, mean_emitted_tokens=2.)
    acceptance = {(model, layout): {'cases': {
        case: {'acceptance': {'grammar_constrained' if case == 'structured-json-schema'
                             else 'unconstrained': copy.deepcopy(rate)}}
        for case in RENDER['CASES'] if case != 'counting'}}
        for model in ('full', 'exl3') for layout in ('single', 'dual')}
    text = RENDER['render'](data, acceptance=acceptance)
    section = text.split('**Adaptive draft acceptance by content.**')[1].split('**Deployment')[0]
    assert section.count('50.00% (2.00)') == 36
    assert 'Schema JSON (grammar-constrained)' in section
    assert 'not teacher-forced quant agreement' in section
    acceptance['exl3', 'dual']['cases']['code']['acceptance']['unconstrained']['verified_drafts'] = 0
    with pytest.raises(AssertionError):
        RENDER['render'](data, acceptance=acceptance)


def test_acceptance_trace_settlement_matches_each_scheduler_path():
    collector = runpy.run_path(str(ROOT/'scripts/collect-ds41-content-acceptance.py'))
    policy = runpy.run_path(str(ROOT/'scripts/summarize-ds41-native-policy.py'))
    observations = [
        {'request_id': 1, 'terminal': True},
        {'request_id': 2, 'terminal': False},
        {'request_id': 2, 'terminal': True},
        {'request_id': 3, 'terminal': True},
    ]
    text = '\n'.join([
        'independent snapshot published lane=0 request_id=1',
        'independent snapshot published lane=1 request_id=2',
        'independent snapshot published lane=0 request_id=3',
    ])
    single, single_marker = collector['completion_markers'](
        observations, text, 1, policy['FIELDS'])
    dual, dual_marker = collector['completion_markers'](
        observations, text, 2, policy['FIELDS'])
    assert single == dual == {1, 2, 3}
    assert single_marker == 'settled request traces'
    assert dual_marker == 'published snapshots'


def test_acceptance_loader_reproduces_traces_and_rejects_tampering(tmp_path):
    import hashlib
    data = reports()
    collector = runpy.run_path(str(ROOT/'scripts/collect-ds41-content-acceptance.py'))
    policy = runpy.run_path(str(ROOT/'scripts/summarize-ds41-native-policy.py'))
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    write(tmp_path/'complete.json', {'results': [{'passed': True}] * 4})
    write(tmp_path/'restoration.json', {'errors': []})
    for model in ('full', 'exl3'):
        for count in (1, 2):
            directory = tmp_path/f'{model}-rtx{count}'
            write(directory/'deployment.json', {
                'binary_sha256': data[model]['binary_sha256'],
                'coordinator': {'Config': {'Labels': {
                    'org.opencontainers.image.revision': data[model]['engine_commit']}}}})
            cases = {}
            for case in set(RENDER['CASES']) - {'counting'}:
                thinking = 'enabled' if case == 'code-reasoning' else 'disabled'
                effort = 'high' if thinking == 'enabled' else None
                constrained = 'true' if case == 'structured-json-schema' else 'false'
                raw = '\n'.join(
                    f'native draft policy observation request_id={i} lane=0 generated=10 '
                    f'verifier_rows=3 matched_prefix=1 raw_confidence=[0,0] constrained={constrained} '
                    'eos=false length_limit=false' for i in (1, 2, 3)).encode()
                observations, _ = policy['parse'](raw.decode())
                cases[case] = dict(thinking=thinking, reasoning_effort=effort,
                    trace_sha256=hashlib.sha256(raw).hexdigest(),
                    acceptance=collector['summarize_acceptance'](observations))
                request = {'thinking': {'type': thinking}, 'reasoning_effort': effort}
                write(directory/f'content/{case}.json', {'passed': True, 'samples': [
                    dict(case=case, repeat=i, passed=True, request=request) for i in (1, 2, 3)]})
                (directory/f'content/{case}-trace.log').write_bytes(raw)
            write(directory/'content/summary.json', dict(passed=True, repeats=3, concurrency=1,
                  rtx_gpus=count, draft_policy='adaptive', cases=cases,
                  corpus_sha256=data[model]['corpus_sha256']))
    result = RENDER['load_acceptance'](tmp_path, data)
    assert len(result) == 4
    trace = tmp_path/'exl3-rtx2/content/code-reasoning-trace.log'
    trace.write_bytes(trace.read_bytes().replace(b'matched_prefix=1', b'matched_prefix=2'))
    with pytest.raises(AssertionError):
        RENDER['load_acceptance'](tmp_path, data)


def synthetic_quant_analysis():
    common = {'routed_expert': {'payload_bytes': 200, 'tensors': 1, 'dtypes': {}, 'shapes': []},
              'shared_expert': {'payload_bytes': 2, 'tensors': 1, 'dtypes': {}, 'shapes': []},
              'embedding': {'payload_bytes': 2, 'tensors': 1, 'dtypes': {}, 'shapes': []},
              'head': {'payload_bytes': 2, 'tensors': 1, 'dtypes': {}, 'shapes': []},
              'other': {'payload_bytes': 2, 'tensors': 1, 'dtypes': {}, 'shapes': []}}
    def snapshot(revision, size, ple, ple_shapes):
        categories = copy.deepcopy(common)
        categories['ple'] = {'payload_bytes': ple, 'tensors': len(ple_shapes),
                             'dtypes': {}, 'shapes': ple_shapes}
        return {'revision': revision, 'snapshot_bytes': size, 'shard_count': 52,
                'categories': categories}
    full_revision = 'a' * 40
    full = snapshot(full_revision, 1000, 300, [{'dtype':'F8_E4M3','shape':[10,256],'tensors':2}])
    exl3 = snapshot('b'*40, 850, 300, [
        {'dtype':'F8_E4M3','shape':[10,256],'tensors':2},
        {'dtype':'F8_E8M0','shape':[10,8],'tensors':2}])
    fp4 = snapshot('c'*40, 700, 150, [
        {'dtype':'F32','shape':[],'tensors':2},
        {'dtype':'F8_E4M3','shape':[10,16],'tensors':2},
        {'dtype':'U8','shape':[10,128],'tensors':2}])
    tiers=[]; shapes=[]
    counts={'w1':(13530,2214),'w2':(9840,5904),'w3':(12054,3690)}
    dims={'w1':([5120],[2304]),'w2':([2304],[5120]),'w3':([5120],[2304])}
    for projection,(low,high) in counts.items():
        for bits,count in [(3,low),(4,high)]:
            tiers.append({'scope':'layers','projection':projection,'bits':bits,'tensors':count,
                          'logical_weights':count*5120*2304,'stored_bytes':count})
            shapes.append({'projection':projection,'bits':bits,'suh':dims[projection][0],
                           'svh':dims[projection][1],'trellis':[1,1,bits*16],'tensors':count})
    return {'schema':1,'passed':True,'snapshots':{'full':full,'exl3':exl3,'exl3_fp4ple':fp4},
            'exl3':{'metadata':{'provenance':{'source_revision':full_revision}},
                    'nominal_average_bpw':3.25,'packed_effective_bpw':3.260072,
                    'tensor_entries':47232,'tiers':tiers,'shapes':shapes},
            'hardlink_clone':{'shared_shards':48,'changed_shards':['49','50','51','52']}}


def synthetic_top1():
    categories = {f'category-{i:02d}': {'samples':12,'matches':11,'agreement':11/12,
        'wilson_95':[.646,.985], 'baseline_token_count':10,'candidate_token_count':10}
        for i in range(11)}
    reports={'full':{'categories':None},
             'exl3':{'overall':{'samples':132,'matches':121,'agreement':121/132,
                 'wilson_95':[.856,.954]},'categories':copy.deepcopy(categories)},
             'fp4ple':{'overall':{'samples':132,'matches':120,'agreement':120/132,
                 'wilson_95':[.846,.949]},'categories':copy.deepcopy(categories)}}
    results={key:{'launch_seconds':60.,'collection_seconds':150.} for key in reports}
    return {'complete':{'corpus_sha256':'d'*64},'reports':reports,'results':results}


def test_quant_and_top1_sections_render_sizes_tiers_denominators_and_protocol():
    text = RENDER['render'](reports(), quant=synthetic_quant_analysis(), top1=synthetic_top1())
    section = text.split('### Quant analysis')[1]
    assert '**Checkpoint and tensor payload sizes.**' in section
    assert '**EXL3 routed projection tiers.**' in section and '3.2601 bpw' in section
    assert '**PLE table geometry.**' in section and '48 unchanged shards' in section
    assert '**Fixed-history top-1 agreement.**' in section
    assert 'same 20-layer TP2 placement' in section and 'exact token-ID denominators' in section
    assert section.count('11/12 (91.7%)') == 22


def materialize_top1(directory):
    import hashlib
    collector = runpy.run_path(str(ROOT/'scripts/collect-ds41-top1-agreement.py'))
    corpus_hash='e'*64;binary='f'*64
    ids=[(f'category-{i//12:02d}-{i:03d}',f'category-{i//12:02d}') for i in range(132)]
    reports={};results=[]
    baseline_sha=None
    for case in ('full','exl3','fp4ple'):
        target=directory/case;target.mkdir(parents=True)
        trace=bytearray();samples=[]
        for index,(identifier,category) in enumerate(ids,1):
            token=index%97
            if case!='full' and index%13==0:token+=1
            line=(f'INFO ds41rt::top1_agreement: native prompt top-1 request_id={index} '
                  f'prompt_tokens={100+index} token_id={token}\n').encode()
            start=len(trace);trace.extend(line)
            reference=index%97
            samples.append({'id':identifier,'category':category,'token_id':token,
                'reference_token_id':reference,'matches_reference':token==reference,
                'request_id':index,'prompt_tokens':100+index,'trace_start':start,
                'trace_end':len(trace),'trace_sha256':hashlib.sha256(line).hexdigest()})
        (target/'trace.log').write_bytes(trace)
        report={'passed':True,'role':'baseline' if case=='full' else 'candidate',
                'binary_sha256':binary,'corpus_sha256':corpus_hash,
                'checkpoint_revision':case,'protocol':{
                    'target_only':True,'draft_model':'disabled','concurrency':1,'temperature':0,
                    'top_p':1,'max_tokens':1,'thinking':'disabled',
                    'token_source':'instrumented target-model scores.select(mask) after fixed-prompt prefill',
                    'constrained':False,'comparison':'exact token ID on byte-identical OpenAI messages'},
                'samples':samples,'overall':None,'categories':None,'reference':None}
        if case!='full':
            report['reference']={'sha256':baseline_sha}
            report['overall']=collector['summary'](samples)
            report['categories']={category:collector['summary']([row for row in samples if row['category']==category])
                                  for _,category in ids[::12]}
        path=target/'top1.json';path.write_text(json.dumps(report))
        if case=='full':baseline_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        results.append({'case':case,'passed':True,'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'launch_seconds':1.,'collection_seconds':2.})
        reports[case]=report
    (directory/'complete.json').write_text(json.dumps({'results':results,'corpus_sha256':corpus_hash,
        'diagnostic':{'binary_sha256':binary}}))
    (directory/'restoration.json').write_text(json.dumps({'errors':[]}))


def test_top1_loader_reproduces_exact_trace_ids_and_rejects_tampering(tmp_path):
    materialize_top1(tmp_path)
    result=RENDER['load_top1'](tmp_path)
    assert result['reports']['exl3']['overall']['samples']==132
    trace=tmp_path/'fp4ple/trace.log'
    trace.write_bytes(trace.read_bytes().replace(b'token_id=1\n',b'token_id=2\n',1))
    with pytest.raises(AssertionError):
        RENDER['load_top1'](tmp_path)
