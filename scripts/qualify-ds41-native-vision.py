#!/usr/bin/env python3
"""Live image semantics, identity isolation, replay and exact-hit qualification.

Run endpoints serially: they share the four Spark workers. Supply the mountain
10016.jpg and Baidu baidu.png fixtures from glmrt-release's Ovis test directory.
The report records fixture hashes, not the external image bytes. Longer greedy
descriptions may differ between target and dSpark; this checks semantic content
and exact repetition within each endpoint, not cross-mode bitwise equivalence.
"""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--fixtures-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--session', default='native-vision-live-v1')
    args = parser.parse_args()
    assert not args.output.exists(), 'choose a new output path to preserve evidence'
    paths = [args.fixtures_dir / name for name in ['10016.jpg', 'baidu.png']]
    paths.append(Path(__file__).resolve().parents[1] /
                 'rust/crates/ds41rt-api/src/native_v41/fixtures/black.png')
    images = ['data:image/' + ('jpeg' if p.suffix == '.jpg' else 'png') +
              ';base64,' + base64.b64encode(p.read_bytes()).decode() for p in paths]
    report = dict(base_url=args.base_url, fixtures=[dict(path=str(p),
                  sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
                  cases=[], passed=False)

    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    def call(name, body, contains):
        request = copy.deepcopy(body)
        for message in request['messages']:
            if isinstance(message['content'], list):
                for part in message['content']:
                    if part['type'] == 'image_url':
                        part['image_url']['url'] = 'fixture:' + str(images.index(part['image_url']['url']))
        record = dict(name=name, request=request, expected_ordered_terms=contains)
        report['cases'].append(record)
        save()
        started = time.monotonic()
        req = urllib.request.Request(args.base_url + '/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=240) as response:
            record['response'] = result = json.load(response)
        record['elapsed_seconds'] = time.monotonic() - started
        save()
        text = result['choices'][0]['message']['content'].lower()
        assert result['choices'][0]['finish_reason'] == 'stop', record
        offset = 0
        for word in contains:
            alternatives = (word,) if isinstance(word, str) else word
            matches = [(text.find(term, offset), term) for term in alternatives
                       if text.find(term, offset) >= offset]
            assert matches, (name, contains, text)
            found, term = min(matches)
            offset = found + len(term)
        print(json.dumps(dict(name=name, text=text, usage=result['usage']), ensure_ascii=False), flush=True)
        return result

    def body(indices, question, session=None):
        return dict(model='deepseek-ai/DeepSeek-V4.1-Flash', messages=[dict(role='user', content=[
            dict(type='text', text=(session or args.session) + '. ' + question)] + [
            dict(type='image_url', image_url=dict(url=images[i])) for i in indices])],
            thinking=dict(type='disabled'), temperature=0, max_tokens=96)

    def exact(first, repeated):
        assert first['choices'][0]['message'] == repeated['choices'][0]['message']
        usage = repeated['usage']
        assert usage['prompt_cache_hit_tokens'] == usage['prompt_tokens']

    question = 'Describe the main subject of this image in one sentence.'
    seed_body = body([0], question)
    seed = call('mountain', seed_body, ['snow', 'mountain'])
    exact(seed, call('mountain-exact', seed_body, ['snow', 'mountain']))
    call('changed-image-logo', body([1], question), ['baidu'])
    # The question asks for a subject, not a color; a solid black image is blank.
    call('changed-image-black', body([2], question), [('black', 'blank')])
    question = 'Describe the first image and then the second image, in that order, in two short sentences.'
    call('mountain-logo', body([0, 1], question), ['mountain', 'baidu'])
    partial = call('mountain-black-partial', body([0, 2], question), ['mountain', ('black', 'blank')])
    assert 0 < partial['usage']['prompt_cache_hit_tokens'] < partial['usage']['prompt_tokens']
    call('logo-mountain-reordered', body([1, 0], question), ['baidu', 'mountain'])
    follow = copy.deepcopy(seed_body)
    follow['messages'] += [dict(role='assistant', content=seed['choices'][0]['message']['content']),
        dict(role='user', content='What color covers most of the mountain? Answer in one word.')]
    resumed = call('completed-turn-resume', follow, ['white'])
    assert resumed['usage']['prompt_cache_hit_tokens'] >= seed['usage']['total_tokens'] - 1
    sixteen = body([2] * 16, 'What color fills all of these images? Answer in one word.',
                   'sixteen-' + args.session)
    first = call('sixteen', sixteen, ['black'])
    exact(first, call('sixteen-exact', sixteen, ['black']))
    report['passed'] = True
    save()


if __name__ == '__main__':
    main()
