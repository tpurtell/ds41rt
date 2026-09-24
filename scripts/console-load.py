#!/usr/bin/env python3
"""Drive varied traffic at a DS41RT server so the live console at `/` has something to show.

Each worker loops over a mix of request kinds: prose, code, a JSON-schema response,
a tool call (both decoded under a grammar) and an occasional long prompt that
exercises chunked prefill. Streams are read to completion and discarded.

    python3 scripts/console-load.py --workers 6 --minutes 10
"""
import argparse
import http.client
import json
import random
import threading
import time
import urllib.parse

PROSE = [
    "Explain how speculative decoding with a small draft model speeds up autoregressive generation. Use two short paragraphs.",
    "Write a short story about a lighthouse keeper who discovers the light is sending messages.",
    "Summarize the causes and consequences of the 1929 stock market crash for a high school student.",
    "Describe the water cycle, then explain how climate change alters it.",
]
CODE = [
    "Write a Python function that parses an ISO 8601 duration like P3DT4H5M into seconds, with tests.",
    "Implement a lock-free single-producer single-consumer ring buffer in Rust with documentation comments.",
    "Write a TypeScript debounce utility with cancellation and a leading-edge option, plus usage examples.",
    "Write a C function that computes a CRC32 table at startup and checksums a buffer.",
]
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "city_report",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
                "population": {"type": "integer"},
                "landmarks": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
            },
            "required": ["city", "country", "population", "landmarks", "summary"],
            "additionalProperties": False,
        },
    },
}
TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_code",
        "description": "Search a repository for code matching a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
}]
FILLER = ("The coordinator owns attention, routing, shared experts and sampling, while four Spark "
          "ranks execute routed expert slices. ") * 40


def vary(text, rng):
    """Break exact prefix hits for most requests; keep some shared prefixes for partial hits."""
    tag = f"[ticket {rng.randrange(10**6):06d}]"
    return f"{tag} {text}" if rng.random() < 0.6 else f"{text} {tag}"


def request_body(kind, model, rng=random):
    body = {"model": model, "stream": True, "temperature": rng.choice([0, 0, 0.7])}
    if kind == "prose":
        body.update(messages=[{"role": "user", "content": vary(random.choice(PROSE), rng)}], max_tokens=random.choice([256, 512, 900]),
                    thinking={"type": "disabled"})
    elif kind == "code":
        body.update(messages=[{"role": "user", "content": vary(random.choice(CODE), rng)}], max_tokens=random.choice([512, 1024, 1500]),
                    thinking={"type": "disabled"})
    elif kind == "reasoning":
        body.update(messages=[{"role": "user", "content": vary(random.choice(CODE), rng)}], max_tokens=1200)
    elif kind == "schema":
        city = random.choice(["Taipei", "Lisbon", "Nairobi", "Montreal", "Osaka"])
        body.update(messages=[{"role": "user", "content": f"Report on {city} as JSON."}], max_tokens=300,
                    response_format=SCHEMA, thinking={"type": "disabled"})
    elif kind == "tool":
        body.update(messages=[{"role": "user", "content": "Find where the scheduler retires finished requests, then explain it."}],
                    tools=TOOLS, tool_choice="required", max_tokens=200, thinking={"type": "disabled"})
    elif kind == "long":
        repeat = random.choice([3, 6, 10])
        body.update(messages=[{"role": "user", "content": vary(FILLER * repeat + "\nIn three sentences, what does the text say?", rng)}],
                    max_tokens=200, thinking={"type": "disabled"})
    return body


def run_one(url, kind, model, rng=random):
    parsed = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=600)
    data = json.dumps(request_body(kind, model, rng))
    conn.request("POST", "/v1/chat/completions", body=data, headers={"content-type": "application/json"})
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"{kind}: HTTP {response.status} {response.read()[:200]!r}")
    while response.read(4096):
        pass
    conn.close()


def worker(index, args, stop, counts, lock):
    kinds = ["prose"] * 3 + ["code"] * 4 + ["reasoning"] + ["schema"] * 2 + ["tool"] * 2 + ["long"]
    rng = random.Random(index)
    time.sleep(index * args.stagger)
    while not stop.is_set():
        kind = rng.choice(kinds)
        try:
            run_one(args.url, kind, args.model, rng)
            with lock:
                counts[kind] = counts.get(kind, 0) + 1
        except Exception as error:  # keep driving load through transient failures
            with lock:
                counts["errors"] = counts.get("errors", 0) + 1
            print(f"worker {index}: {error}", flush=True)
            time.sleep(1)
        time.sleep(rng.uniform(0, args.pause))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--stagger", type=float, default=2.0, help="seconds between worker starts")
    parser.add_argument("--pause", type=float, default=2.0, help="max idle seconds between a worker's requests")
    args = parser.parse_args()
    stop, lock, counts = threading.Event(), threading.Lock(), {}
    threads = [threading.Thread(target=worker, args=(i, args, stop, counts, lock), daemon=True) for i in range(args.workers)]
    for thread in threads:
        thread.start()
    deadline = time.time() + args.minutes * 60
    try:
        while time.time() < deadline:
            time.sleep(10)
            with lock:
                print(time.strftime("%H:%M:%S"), json.dumps(counts, sort_keys=True), flush=True)
    except KeyboardInterrupt:
        pass
    stop.set()


if __name__ == "__main__":
    main()
