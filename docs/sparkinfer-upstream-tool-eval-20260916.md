# Upstream integration: high-thinking tool evaluation

Three clean dual-RTX dSpark campaigns completed all 264 scenarios. Each used
C16, high thinking, temperature zero, a 900-second timeout, twelve tool turns,
and the benchmark's **4096-token per-response cap, including reasoning**.
No scenario or campaign was retried.

| Run | Basic | Hard | Total | Pass / partial / fail |
|---:|---:|---:|---:|---:|
| 1 | 124/138 | 31/38 | 155/176 | 69 / 17 / 2 |
| 2 | 126/138 | 34/38 | 160/176 | 73 / 14 / 1 |
| 3 | 123/138 | 33/38 | 156/176 | 72 / 12 / 4 |
| Mean | 124.33/138 | 32.67/38 | 157.00/176 | — |

The installed benchmark always passes `max_tokens=4096` unless overridden.
The original wrapper mistakenly described an omitted override as the server's
output policy. The raw summaries preserve that original null override; this
report and the wrapper now distinguish it from the effective cap. In run 3,
TC-88 returned no visible first answer after lengthy reasoning, then returned
two 20-digit follow-up answers. This is consistent with exhausting the first
response's reasoning/output budget; the trace does not retain its finish reason.
It remains a scored failure, with no replacement run.

Zero-score outcomes are retained below; partial scores and all traces remain
in the evidence archive.

- Run 1, TC-43: Called web_search with an empty query — violated required parameter constraint.
- Run 1, TC-74: Sent an unsafe, duplicate, or premature confirmation email.
- Run 2, TC-43: Called web_search with an empty query — violated required parameter constraint.
- Run 3, TC-43: Called web_search with an empty query — violated required parameter constraint.
- Run 3, TC-61: Did not attempt to run the analysis script.
- Run 3, TC-68: Called tools when none were needed.
- Run 3, TC-88: Returned extra text or a value that was not exactly 20 digits.

The [machine-readable report](sparkinfer-upstream-tool-eval-20260916.json)
records exact image/source identity, arguments, hardware settings, benchmark
version, and file hashes. The [complete evidence archive](evidence/upstream-clean-tools-20260916.tar.gz) preserves each command, result,
trace, benchmark database, and log. The compatibility adapter is called vllm;
the actual serving engine is native DS41RT.
