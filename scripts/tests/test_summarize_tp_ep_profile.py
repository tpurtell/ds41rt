"""CPU-only regression tests for the matched TP×EP decode-profile parser.

Fixtures are verbatim raw lines captured from
`runs/tp-ep-preflight/g2-live/profile/tp2/` (ANSI-wrapped lines included), so the
parser is validated against the actual emitter format, not invented fields. No
GPU, container, service or build operation.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "summarize-ds41-tp-ep-profile.py"


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_ds41_tp_ep_profile", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()

# Verbatim captured lines (ANSI stripped except where the raw form is kept).
WORKER_ROWS64 = (
    "2026-09-20T08:22:00.186850Z DEBUG ds41rt::expert_timing: native expert execution "
    "layer=20 executor_id=17 rows=64 active_experts=16 direct_registered_output=true "
    "kernel_capacity=80 max_expert_rows=34 expert_rows_histogram=[8, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 2] "
    "expert_rows_tail_routes=56 unique_expert_weight_bytes=150405120 output_bytes=655360 "
    "upload_us=36 kernel_us=865.343994140625 compact_us=26.33599853515625 "
    "execution_host_us=897 download_us=0 total_us=933"
)
WORKER_ROWS128 = (
    "2026-09-20T08:21:12.387082Z DEBUG ds41rt::expert_timing: native expert execution "
    "layer=20 executor_id=17 rows=128 active_experts=38 direct_registered_output=true "
    "kernel_capacity=4096 max_expert_rows=71 expert_rows_histogram=[7, 2, 2, 1, 5, 3, 2, 2, 3, 0, 0, 1, 0, 1, 0, 1, 8] "
    "expert_rows_tail_routes=258 unique_expert_weight_bytes=357212160 output_bytes=1310720 "
    "upload_us=826 kernel_us=1672.1279296875 compact_us=0 execution_host_us=0 download_us=0 total_us=0"
)
# Raw ANSI-wrapped form of a real coordinator line.
TARGET_EXPERTS_ANSI = (
    "\x1b[2m2026-09-20T08:21:12.757941Z\x1b[0m \x1b[34mDEBUG\x1b[0m "
    "\x1b[2mds41rt::timing\x1b[0m\x1b[2m:\x1b[0m target experts with TP2 shared "
    "\x1b[3mlayer\x1b[0m\x1b[2m=\x1b[0m20 \x1b[3mrows\x1b[0m\x1b[2m=\x1b[0m128 "
    "\x1b[3mrouted_us\x1b[0m\x1b[2m=\x1b[0m240 \x1b[3mdispatch_us\x1b[0m\x1b[2m=\x1b[0m493205 "
    "\x1b[3mshared_and_collect_us\x1b[0m\x1b[2m=\x1b[0m3984"
)
TARGET_COLLECTION = (
    "2026-09-20T08:21:12.757928Z DEBUG ds41rt::timing: target collection layer=20 rows=128 "
    "shared_copy_us=0 upload_us=18 receive_us=3653 reduce_us=43"
)
LANE_ROUND = (
    "2026-09-20T08:21:54.303336Z DEBUG ds41rt::timing: native independent lane round lane=0 "
    "requests=1 proposed=7 accepted=1 emitted=1 draft_us=6371 prepared_us=6392 "
    "verify_us=46335 total_us=53563"
)
TRANSPORT = (
    "protocol_v2_verbs_persistent_server_roundtrip_timing execution_lane=0 request_id=1 "
    "layer_id=20 rows=128 routes=768 request_wire_bytes=690272 response_frames=1 "
    "direct_device_response_frames=1 response_wire_bytes=1310816 executor=local-owner "
    "poll_recv_ms=0.006 copy_header_ms=0.000 resize_ms=0.000 copy_request_ms=0.000 "
    "post_recv_ms=0.001 parse_ms=0.004 execute_ms=2.665 encode_ms=0.001 send_ms=0.003 "
    "poll_send_ms=0.000 total_ms=2.720"
)


def summarize(lines, tp, **kwargs):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "profile.log"
        path.write_text("\n".join(lines) + "\n")
        return MODULE.summarize([path], tp, **kwargs)


def summarize_files(files, tp, **kwargs):
    with tempfile.TemporaryDirectory() as temporary:
        paths = []
        for name, lines in files.items():
            path = Path(temporary) / name
            path.write_text("\n".join(lines) + "\n")
            paths.append(path)
        return MODULE.summarize(paths, tp, **kwargs)


class ProfileParserTest(unittest.TestCase):
    def test_real_worker_line_owned_rows_and_shard_normalization(self):
        record = summarize([WORKER_ROWS64], tp=2)["raw"]["workers"][0]
        # sum((i+1)*bins[:16]) + tail = 74 + 56
        self.assertEqual(record["owned_rows"], 130)
        self.assertEqual(record["tail_routes"], 56)
        self.assertEqual(record["active_experts"], 16)
        self.assertAlmostEqual(record["kernel_us_per_logical_unit"], 865.343994140625 / (130 * 1152))
        # TP2 takes no padding, so both variants agree.
        self.assertAlmostEqual(record["kernel_us_per_kernel_unit"], record["kernel_us_per_logical_unit"])

    def test_tp4_padding_variant_uses_the_kernel_extent(self):
        record = summarize([WORKER_ROWS64], tp=4)["raw"]["workers"][0]
        self.assertAlmostEqual(record["kernel_us_per_logical_unit"], 865.343994140625 / (130 * 576))
        self.assertAlmostEqual(record["kernel_us_per_kernel_unit"], 865.343994140625 / (130 * 640))

    def test_worst_rank_median_uses_each_metric_own_rank(self):
        other = WORKER_ROWS64.replace("executor_id=17", "executor_id=18").replace(
            "kernel_us=865.343994140625", "kernel_us=100.0").replace(
            "compact_us=26.33599853515625", "compact_us=90.0")
        entry = summarize([WORKER_ROWS64, other], tp=2)["worst_rank_median"][0]
        self.assertEqual(entry["kernel_worst_rank"], 17)
        self.assertEqual(entry["compact_worst_rank"], 18)
        self.assertAlmostEqual(entry["worst_rank_median_compact_us"], 90.0)
        self.assertAlmostEqual(entry["kernel_worst_rank_compact_us"], 26.33599853515625)

    def test_ansi_wrapped_target_experts_line_parses_shared_and_collect(self):
        phases = summarize([TARGET_EXPERTS_ANSI], tp=2)["coordinator_phases"]
        self.assertAlmostEqual(phases["routed_us"][0]["median"], 240.0)
        self.assertAlmostEqual(phases["dispatch_us"][0]["median"], 493205.0)
        self.assertAlmostEqual(phases["shared_and_collect_us"][0]["median"], 3984.0)

    def test_nested_phases_stay_separate(self):
        phases = summarize([LANE_ROUND, TARGET_COLLECTION], tp=2)["coordinator_phases"]
        self.assertIn("receive_us", phases)
        self.assertIn("reduce_us", phases)
        self.assertNotIn("total_us", phases)  # never summed

    def test_dspark_math_does_not_double_count(self):
        dspark = summarize([LANE_ROUND], tp=2)["dspark"]
        # accepted=1 is the anchor; requests=1 => zero drafts accepted.
        self.assertEqual(dspark["accepted_inputs_incl_anchor"], 1)
        self.assertEqual(dspark["drafts_accepted"], 0)
        self.assertEqual(dspark["acceptance_ratio"], 0.0)
        self.assertEqual(dspark["emitted_tokens"], 1)
        # tokens/round is emitted only, never accepted+emitted (=2).
        self.assertEqual(dspark["tokens_per_round"], 1.0)

    def test_rows_filter_excludes_other_workloads(self):
        result = summarize([WORKER_ROWS64, WORKER_ROWS128], tp=2, rows_filter={64})
        rows = {r["rows"] for r in result["raw"]["workers"]}
        self.assertEqual(rows, {64})

    def test_rowless_lane_round_survives_the_rows_filter(self):
        result = summarize([WORKER_ROWS64, LANE_ROUND], tp=2, rows_filter={64})
        self.assertEqual(len(result["raw"]["workers"]), 1)
        self.assertEqual(len(result["raw"]["coordinator"]), 1)
        self.assertEqual(result["raw"]["coordinator"][0]["kind"], "lane_round")
        self.assertIsNotNone(result["dspark"])

    def test_no_raw_output_omits_records(self):
        result = summarize([WORKER_ROWS64], tp=2, include_raw=False)
        self.assertNotIn("raw", result)
        self.assertEqual(result["counts"]["workers"], 1)

    def test_transport_defaults_to_per_rank_observation(self):
        result = summarize([TRANSPORT], tp=2)
        self.assertEqual(result["transport"]["mode"], "per_rank_observation")
        self.assertNotIn("worst_rank_total_ms", result["transport"])
        self.assertEqual(result["transport"]["by_rank_rows"][0]["rows"], 128)
        self.assertAlmostEqual(result["transport"]["by_rank_rows"][0]["median"], 2.720)
        self.assertIn("unverified assumption", result["matchability"]["transport_cross_rank"])

    def test_cross_rank_join_requires_run_id_and_flag(self):
        files = {"ostrich.log": [TRANSPORT], "dodo.log": [TRANSPORT.replace("total_ms=2.720", "total_ms=3.100")]}
        # Without the explicit opt-in the parser must not claim a cross-rank join.
        self.assertEqual(summarize_files(files, tp=2)["transport"]["mode"], "per_rank_observation")
        joined = summarize_files(files, tp=2, run_id="run42", assume_global_request_id=True)
        self.assertEqual(joined["transport"]["mode"], "assumed_global_ids_unverified")
        entry = joined["transport"]["requests"][0]
        self.assertEqual(entry["ranks"], 2)
        self.assertEqual(entry["worst_rank"], "dodo.log")
        self.assertAlmostEqual(entry["worst_rank_total_ms"], 3.100)

    def test_cross_rank_excludes_missing_ids(self):
        unjoinable = TRANSPORT.replace("request_id=1 ", "")
        result = summarize_files({"a.log": [unjoinable], "b.log": [TRANSPORT]}, tp=2,
                                 run_id="run42", assume_global_request_id=True)
        self.assertEqual(result["matchability"]["transport_unjoinable_records"], 1)
        self.assertEqual(len(result["transport"]["requests"]), 1)

    def test_matchability_and_raw_retention(self):
        result = summarize([WORKER_ROWS64, TARGET_EXPERTS_ANSI, LANE_ROUND, TRANSPORT], tp=2)
        matchability = result["matchability"]
        self.assertIn("no request id", matchability["worker_coordinator_join"])
        self.assertIn("not a per-request critical path", matchability["worker_aggregate"])
        self.assertIn("binned", matchability["histogram_kind"])
        self.assertIn("resident-footprint", matchability["weight_footprint"])
        self.assertEqual(len(result["raw"]["workers"]), 1)
        self.assertEqual(len(result["raw"]["coordinator"]), 2)
        self.assertEqual(len(result["raw"]["transport"]), 1)
        self.assertIn("draft", result["scope"])


if __name__ == "__main__":
    unittest.main()
