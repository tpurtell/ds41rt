"""CPU-only coverage for the opt-in Spark TP x EP replicated topology.

No Docker, SSH, GPU, build or serving operation is performed. Every case drives
the launcher through `scripts/release-common.sh` or the process-boundary stubs
used by `test_placement_handoff.py`.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ds41rt.config"
EXAMPLES = ROOT / "examples" / "configs"


def source_common(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-euc", "source scripts/release-common.sh; " + script, "test", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def load(extra: str = "", command: str = 'printf "%s\\n" "$SPARK_COUNT" "$SPARK_TP" "$SPARK_EP"') -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as temporary:
        config = Path(temporary) / "topology.config"
        config.write_text(CONFIG.read_text() + "\n" + extra + "\n")
        return source_common('release_load_config "$1"; ' + command, str(config))


def load_file(path: Path, command: str = 'printf "%s\\n" "$SPARK_COUNT" "$SPARK_TP" "$SPARK_EP"') -> subprocess.CompletedProcess[str]:
    return source_common('release_load_config "$1"; ' + command, str(path))


# Six placeholders are deliberate: REPLACE-ME-* is not resolvable and
# 192.0.2.0/24 is the RFC 5737 documentation range. The base config's legacy
# four-rank secondary rail is cleared so the six-rank rail stays all-or-none.
SIX_HOSTS = """\
SPARK_4_HOST=REPLACE-ME-spark-4
SPARK_5_HOST=REPLACE-ME-spark-5
SPARK_4_LANE_A=192.0.2.5
SPARK_5_LANE_A=192.0.2.6
SPARK_0_LANE_B=
SPARK_1_LANE_B=
SPARK_2_LANE_B=
SPARK_3_LANE_B=
"""


class TopologyConfigTest(unittest.TestCase):
    def test_default_config_keeps_legacy_topology_and_is_not_mutated(self) -> None:
        before = CONFIG.read_bytes()
        result = load(
            "",
            'printf "%s\\n" "$SPARK_COUNT" "$SPARK_TP" "$SPARK_EP" '
            '"$(release_spark_tp)" "$(release_spark_ep)"; '
            "if release_spark_topology_explicit; then echo explicit; else echo legacy; fi",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["4", "", "", "4", "1", "legacy"])
        self.assertEqual(CONFIG.read_bytes(), before)
        self.assertNotIn("SPARK_TP=", CONFIG.read_text())
        self.assertNotIn("SPARK_EP=", CONFIG.read_text())

    def test_approved_topologies_resolve_rank_map(self) -> None:
        cases = {
            "SPARK_TP=2\nSPARK_EP=2\n": ["0 0 0", "1 0 1", "2 1 0", "3 1 1"],
            "SPARK_COUNT=6\nSPARK_TP=3\nSPARK_EP=2\nRTX_GPUS=1\n" + SIX_HOSTS: [
                "0 0 0", "1 0 1", "2 0 2", "3 1 0", "4 1 1", "5 1 2",
            ],
            "SPARK_COUNT=6\nSPARK_TP=2\nSPARK_EP=3\n" + SIX_HOSTS: [
                "0 0 0", "1 0 1", "2 1 0", "3 1 1", "4 2 0", "5 2 1",
            ],
            "SPARK_TP=4\nSPARK_EP=1\n": ["0 0 0", "1 0 1", "2 0 2", "3 0 3"],
        }
        for overlay, expected in cases.items():
            with self.subTest(overlay=overlay):
                result = load(overlay, "release_spark_rank_map")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), expected)

    def test_six_host_bounds_generate_every_rank_and_peer(self) -> None:
        result = load(
            "SPARK_COUNT=6\nSPARK_TP=3\nSPARK_EP=2\nRTX_GPUS=1\n" + SIX_HOSTS,
            'release_hosts_csv; release_lane_a_csv; release_expert_hosts_csv',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        hosts, lanes, experts = result.stdout.splitlines()
        self.assertEqual(
            hosts,
            "ostrich,dodo,emu,kiwi,REPLACE-ME-spark-4,REPLACE-ME-spark-5",
        )
        self.assertEqual(lanes, "10.55.0.1,10.55.0.2,10.55.0.3,10.55.0.4,192.0.2.5,192.0.2.6")
        self.assertTrue(experts.endswith("spark-4=192.0.2.5:19441,spark-5=192.0.2.6:19441"))

    def test_optional_secondary_rail_is_all_or_none_for_six_hosts(self) -> None:
        base = "SPARK_COUNT=6\nSPARK_TP=3\nSPARK_EP=2\nRTX_GPUS=1\n" + SIX_HOSTS
        partial = base + "\n".join(
            f"SPARK_{index}_LANE_B=10.55.0.{index + 5}" for index in range(5)
        )
        result = load(partial, "true")
        self.assertEqual(result.returncode, 2)
        self.assertIn("secondary Spark rail", result.stderr)

        complete = base + "\n".join(
            f"SPARK_{index}_LANE_B=10.55.0.{index + 5}" for index in range(6)
        )
        result = load(complete, "release_lane_b_csv")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.strip().split(",")), 6)

        result = load(base, "release_lane_b_csv")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")

    def test_invalid_combinations_are_rejected(self) -> None:
        cases = {
            "SPARK_TP=2\nSPARK_EP=1\n": "must equal SPARK_COUNT",
            "SPARK_TP=5\nSPARK_EP=2\n": "SPARK_TP must be 2, 3, or 4",
            "SPARK_EP=4\n": "SPARK_EP must be 1, 2, or 3",
            "SPARK_TP=2\n": "set together or omitted together",
            "SPARK_COUNT=6\nSPARK_TP=3\nSPARK_EP=3\n" + SIX_HOSTS: "must equal SPARK_COUNT",
            "SPARK_COUNT=4\nSPARK_TP=3\nSPARK_EP=2\n": "must equal SPARK_COUNT",
            "SPARK_COUNT=6\n" + SIX_HOSTS: "SPARK_COUNT=6 requires explicit",
        }
        for overlay, message in cases.items():
            with self.subTest(overlay=overlay):
                result = load(overlay, "true")
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn(message, result.stderr)

    def test_config_accepts_any_rtx_gpus_and_defers_to_the_budget(self) -> None:
        # The Spark TP/EP degree must not hardcode the RTX layout. Both RTX
        # counts parse; the launcher rejects an infeasible combination with the
        # resolved weight budget instead.
        for overlay in ("SPARK_TP=2\nSPARK_EP=2\nRTX_GPUS=1\n",
                        "SPARK_TP=2\nSPARK_EP=2\nRTX_GPUS=2\n",
                        "SPARK_COUNT=6\nSPARK_TP=3\nSPARK_EP=2\nRTX_GPUS=2\n" + SIX_HOSTS,
                        "SPARK_COUNT=6\nSPARK_TP=3\nSPARK_EP=2\nRTX_GPUS=1\n" + SIX_HOSTS):
            with self.subTest(overlay=overlay):
                result = load(overlay, "true")
                self.assertEqual(result.returncode, 0, result.stderr)
        # A single RTX has no local routed layer, so TP2 remote weights overflow
        # the default 100 GiB budget and the budget gate must reject them.
        first_layer = source_common('release_spark_first_layer 1 auto')
        self.assertEqual(first_layer.stdout.strip(), "0")
        self.assertEqual(
            source_common('release_validate_spark_weight_admission 0 2 107374182400').returncode,
            2,
        )

    def test_explicit_topology_rejects_non_native_formats(self) -> None:
        for overlay, message in (
            ("EXPERT_FORMAT=exl3\nSPARKINFER_EXL3=auto\n", "requires EXPERT_FORMAT=native"),
            ("EXL3_PAIRED_TP4=on\n", "EXL3_PAIRED_TP4 must be off"),
            ("SPARKINFER_EXL3=auto\n", "requires SPARKINFER_EXL3=disable"),
        ):
            with self.subTest(overlay=overlay):
                result = load("SPARK_TP=2\nSPARK_EP=2\n" + overlay, "true")
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)

    def test_unknown_topology_alias_is_not_accepted(self) -> None:
        result = load("SPARK_GROUPS=2\n", "true")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown configuration key: SPARK_GROUPS", result.stderr)


class AdmissionTest(unittest.TestCase):
    def report(self, first_layer: int, tp: int, budget: int = 107374182400) -> subprocess.CompletedProcess[str]:
        return source_common(
            'release_validate_spark_weight_admission "$1" "$2" "$3"',
            str(first_layer), str(tp), str(budget),
        )

    def test_exact_layer_bytes(self) -> None:
        expected = {2: "3609722880", 3: "2406481920", 4: "2005401600"}
        for tp, value in expected.items():
            with self.subTest(tp=tp):
                result = source_common('release_spark_layer_bytes "$1"', str(tp))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), value)
        result = source_common('release_spark_layer_bytes 1')
        self.assertEqual(result.returncode, 2)

    def test_weight_only_boundary_thresholds(self) -> None:
        # TP2 remote weights need at least 11 RTX-local layers in the 100 GiB
        # budget: 29 remote layers fit, 30 do not.
        ok = self.report(11, 2)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("remote_layers=29", ok.stdout)
        self.assertIn("workspace_accounted=no", ok.stdout)
        self.assertIn(f"per_rank_weight_bytes={29 * 3609722880}", ok.stdout)

        bad = self.report(10, 2)
        self.assertEqual(bad.returncode, 2)
        self.assertIn("weight-only admission fails", bad.stderr)

        # TP3 can hold every layer in the weight budget; TP4 is padded.
        tp3 = self.report(0, 3)
        self.assertEqual(tp3.returncode, 0, tp3.stderr)
        self.assertIn("remote_layers=40", tp3.stdout)
        self.assertIn(f"per_rank_weight_bytes={40 * 2406481920}", tp3.stdout)

        tp4 = self.report(0, 4)
        self.assertEqual(tp4.returncode, 0, tp4.stderr)
        self.assertIn(f"per_rank_weight_bytes={40 * 2005401600}", tp4.stdout)

    def test_dynamic_boundary_changes_the_verdict(self) -> None:
        # The same topology flips at the real boundary; no hardcoded 20.
        self.assertEqual(self.report(20, 2).returncode, 0)
        self.assertEqual(self.report(0, 2).returncode, 2)


class ExampleConfigTest(unittest.TestCase):
    def test_every_example_parses_with_its_approved_geometry(self) -> None:
        expected = {
            "tp4ep1-explicit-native.config": ("4", "4", "1"),
            "tp2ep2-native.config": ("4", "2", "2"),
            "tp3ep2-native.config": ("6", "3", "2"),
            "tp2ep3-native.config": ("6", "2", "3"),
        }
        self.assertEqual(
            sorted(path.name for path in EXAMPLES.glob("*.config")),
            sorted(expected),
        )
        for name, geometry in expected.items():
            with self.subTest(name=name):
                result = load_file(EXAMPLES / name)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(tuple(result.stdout.splitlines()), geometry)

    def test_six_host_examples_use_the_connected_fifth_and_sixth_sparks(self) -> None:
        for name in ("tp3ep2-native.config", "tp2ep3-native.config"):
            text = (EXAMPLES / name).read_text()
            self.assertIn("SPARK_4_HOST=rhea", text)
            self.assertIn("SPARK_5_HOST=moa", text)
            # rhea/moa fabric A; the candidate launcher is single-rail A-only,
            # so the six-host secondary rail stays unset.
            self.assertIn("SPARK_4_LANE_A=10.55.0.5", text)
            self.assertIn("SPARK_5_LANE_A=10.55.0.6", text)
            self.assertIn("NOT RELEASE-QUALIFIED", text)
            self.assertNotIn("SPARK_4_LANE_B=", text)
            self.assertNotIn("SPARK_5_LANE_B=", text)
            self.assertNotIn("REPLACE-ME", text)
            self.assertNotIn("192.0.2.", text)
        # The default config must not synthesize the fifth/sixth Sparks.
        default = CONFIG.read_text()
        self.assertNotIn("SPARK_4_HOST", default)
        self.assertNotIn("SPARK_5_HOST", default)
        self.assertNotIn("SPARK_4_LANE_A", default)
        self.assertNotIn("SPARK_5_LANE_A", default)

    def test_two_rtx_tp2ep2_example_is_opt_in_and_bound_to_real_hosts(self) -> None:
        text = (EXAMPLES / "tp2ep2-native.config").read_text()
        self.assertIn("SPARK_TP=2", text)
        self.assertIn("SPARK_EP=2", text)
        self.assertIn("RTX_GPUS=2", text)
        self.assertIn("SPARK_0_HOST=ostrich", text)
        self.assertIn("SPARK_3_HOST=kiwi", text)


STUB = r'''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
args=sys.argv[1:]; tool=Path(sys.argv[0]).name
with open(os.environ['EVENTS'],'a') as f:f.write(json.dumps([tool,args])+'\n')
if tool=='docker':
 if args[0]=='inspect':print('running')
 elif args[0]=='exec' and 'cat' in args:
  print(Path(os.environ['PLAN']).read_text())
 elif args[0]=='run':print('container-id')
else:
 sys.stdin.read() if '-s' in args else None
'''


class LauncherTopologyTest(unittest.TestCase):
    """Run run.sh's real startup block with process-boundary stubs."""

    def run_startup(self, *, gpus, plan, topology_explicit, spark_tp, spark_ep, spark_count, hosts,
                    extra_env=None, extra_setup=""):
        source = (ROOT / "run.sh").read_text()
        block = source[source.index("placement_directory="):source.index("api_url=")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan").write_text(json.dumps(plan))
            for name in ["docker", "ssh"]:
                path = root / name
                path.write_text(STUB)
                path.chmod(0o755)
            env = dict(
                os.environ,
                PATH=str(root) + os.pathsep + os.environ["PATH"],
                EVENTS=str(root / "events"),
                PLAN=str(root / "plan"),
                **(extra_env or {}),
            )
            setup = f'''set -euo pipefail
release_die() {{ echo "$*" >&2; exit 1; }}
release_validate_spark_weight_admission() {{ echo "remote_layers=stub"; }}
RELEASE_RTX_GPUS="$1"
coordinator=coordinator
snapshot_rel=model
peers=peer-list
ADDR=127.0.0.1:8000
PREFILL_BATCH_TOKENS=2048
CONCURRENCY=16
PREFIX_CACHE_ENTRIES=20
MAX_CONTEXT_TOKENS=1048576
MAX_OUTPUT_TOKENS=393216
HTTP_QUEUE_WAIT_MS=25000
RTX_EXPERT_LAYERS=auto
HOST_CACHE_BYTES=auto
KV_POOL_SIZE=
MEMORY_RESERVATION=
DSPARK=on
DSPARK_DRAFT_POLICY=adaptive
dspark_draft_limit=
TP2_ATTENTION=off
TP2_QUERY_PROJECTION=off
TP2_OUTPUT_PROJECTION=off
TP2_DSPARK_EXPERTS=off
topology_explicit={topology_explicit}
spark_tp={spark_tp}
spark_ep={spark_ep}
spark_admission=not-applicable
spark_exl3_identity=
gpu_request=device=uuid0,uuid1
gpu_uuid_csv=uuid0,uuid1
fingerprint=test
hf_home=/models
COORDINATOR_DOCKER_INFERENCE=coordinator-image
SPARK_EXPERT_DOCKER_INFERENCE=spark-image
SPARK_DEVICE_BUDGET_BYTES=107374182400
spark_prefix=worker
hosts=({' '.join(hosts)})
EXPERT_PORT=19441
expert_capacity=4096
spark_first_layer=0
SPARK_COUNT={spark_count}
{extra_setup}'''
            result = subprocess.run(
                ["bash", "-c", setup + block, "test", str(gpus)],
                env=env,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
            events_path = root / "events"
            events = (
                [json.loads(line) for line in events_path.read_text().splitlines()]
                if events_path.exists()
                else []
            )
            return result, events

    def test_default_launch_passes_no_topology_flags_or_dummy_peers(self) -> None:
        result, events = self.run_startup(
            gpus=2,
            plan=dict(version=1, rtx_gpus=2, nonce="fresh", rtx_expert_layers=20, spark_first_layer=20),
            topology_explicit=0,
            spark_tp=4,
            spark_ep=1,
            spark_count=4,
            hosts=["a", "b", "c", "d"],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        coordinator = next(args for tool, args in events if tool == "docker" and args[0] == "run")
        self.assertNotIn("--spark-tp", coordinator)
        self.assertNotIn("--spark-ep", coordinator)
        # Worker positional tail ends with topology(explicit,tp,ep) then the
        # three optional RDMA env values.
        self.assertTrue(all(args[-6] == "0" for tool, args in events if tool == "ssh" and "-s" in args))

    def test_explicit_topology_reaches_coordinator_and_every_worker(self) -> None:
        result, events = self.run_startup(
            gpus=2,
            plan=dict(version=1, rtx_gpus=2, nonce="fresh", rtx_expert_layers=20, spark_first_layer=20),
            topology_explicit=1,
            spark_tp=2,
            spark_ep=2,
            spark_count=4,
            hosts=["a", "b", "c", "d"],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        coordinator = next(args for tool, args in events if tool == "docker" and args[0] == "run")
        self.assertEqual(coordinator[coordinator.index("--spark-tp") + 1], "2")
        self.assertEqual(coordinator[coordinator.index("--spark-ep") + 1], "2")
        # The launcher peer list is assembled from the real lane addresses for
        # count 4/6; the zero-Spark dummy list is unreachable for an explicit
        # topology because SPARK_COUNT must be 4 or 6.
        self.assertEqual(coordinator[coordinator.index("--peers") + 1], "peer-list")
        starts = [args for tool, args in events if tool == "ssh" and "-s" in args]
        self.assertEqual(len(starts), 4)
        for args in starts:
            self.assertEqual(args[-6:-3], ["1", "2", "2"])
            self.assertEqual(args[-7], "4")

    def test_six_rank_explicit_topology_starts_six_workers(self) -> None:
        result, events = self.run_startup(
            gpus=2,
            plan=dict(version=1, rtx_gpus=2, nonce="fresh", rtx_expert_layers=12, spark_first_layer=12),
            topology_explicit=1,
            spark_tp=2,
            spark_ep=3,
            spark_count=6,
            hosts=["a", "b", "c", "d", "e", "f"],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        starts = [args for tool, args in events if tool == "ssh" and "-s" in args]
        self.assertEqual(len(starts), 6)
        self.assertEqual([args[-8] for args in starts], ["12"] * 6)
        self.assertEqual([args[-7] for args in starts], ["6"] * 6)
        self.assertEqual([args[-6:-3] for args in starts], [["1", "2", "3"]] * 6)

    def test_worker_remote_block_emits_flags_only_when_explicit(self) -> None:
        source = (ROOT / "run.sh").read_text()
        _, remote = source.split("echo \"== starting native Spark experts ==\"", 1)[1].split("<<'REMOTE' &", 1)
        remote = remote.split("\nREMOTE", 1)[0].replace(" >/dev/null", "")
        harness = 'docker() { printf "%s\\n" "$@"; }\n' + remote
        explicit = subprocess.run(
            ["bash", "-euc", harness, "test", "image", "name", "1", "4096",
             "107374182400", "19441", "snapshot", "fingerprint", "3", "4", "1", "2", "2"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(explicit.returncode, 0, explicit.stderr)
        arguments = explicit.stdout.splitlines()
        self.assertEqual(arguments[arguments.index("--spark-tp") + 1], "2")
        self.assertEqual(arguments[arguments.index("--spark-ep") + 1], "2")
        self.assertEqual(arguments[arguments.index("--rank") + 1], "1")
        self.assertEqual(arguments[arguments.index("--world") + 1], "4")
        self.assertEqual(arguments[arguments.index("--first-layer") + 1], "3")

        # Legacy invocation: ten positional arguments, no topology flags.
        legacy = subprocess.run(
            ["bash", "-euc", harness, "test", "image", "name", "1", "4096",
             "107374182400", "19441", "snapshot", "fingerprint", "3", "4"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(legacy.returncode, 0, legacy.stderr)
        legacy_arguments = legacy.stdout.splitlines()
        self.assertNotIn("--spark-tp", legacy_arguments)
        self.assertEqual(legacy_arguments[legacy_arguments.index("--world") + 1], "4")

    def test_six_rank_launch_forwards_rdma_env_and_draft_controls(self) -> None:
        device_map = "10.55.0.1=rocep1s0f0,10.55.0.6=roceP2p1s0f0"
        result, events = self.run_startup(
            gpus=2,
            plan=dict(version=1, rtx_gpus=2, nonce="fresh", rtx_expert_layers=20, spark_first_layer=20),
            topology_explicit=1,
            spark_tp=2,
            spark_ep=3,
            spark_count=6,
            hosts=["a", "b", "c", "d", "e", "f"],
            extra_env={
                "DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP": device_map,
                "DS41RT_VERBS_APP_IB_PORT_NUM": "1",
                "DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES": "2",
            },
            extra_setup="DSPARK_DRAFT_POLICY=full\ndspark_draft_limit=7\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        coordinator = next(args for tool, args in events if tool == "docker" and args[0] == "run")
        self.assertIn("--dspark-fixed", coordinator)
        self.assertEqual(coordinator[coordinator.index("--dspark-draft-limit") + 1], "7")
        self.assertIn(f"DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP={device_map}", coordinator)
        starts = [args for tool, args in events if tool == "ssh" and "-s" in args]
        self.assertEqual(len(starts), 6)
        for args in starts:
            self.assertEqual(args[-3:], [device_map, "1", "2"])


class VerbsDeviceMapTest(unittest.TestCase):
    """The shared device-map validator is what both launchers depend on."""

    def validate(self, value):
        return subprocess.run(
            ["bash", "-euc", "source scripts/release-common.sh; release_validate_verbs_device_map", "test"],
            cwd=ROOT, text=True, capture_output=True,
            env=dict(os.environ, DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP=value),
        )

    def test_accepts_unique_ipv4_device_entries(self) -> None:
        result = self.validate("10.55.0.1=rocep1s0f0,10.55.0.6=roceP2p1s0f0")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_malformed_or_duplicate_entries(self) -> None:
        for bad in ("10.55.0.1", "10.55.0.1=", "=rocep1s0f0", "10.55.0.1=a,10.55.0.1=b"):
            with self.subTest(bad=bad):
                result = self.validate(bad)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP", result.stderr)


class StopScriptTest(unittest.TestCase):
    """Stop must cover every configured rank, not a fixed four-host list."""

    def test_usage_names_configured_ranks(self) -> None:
        text = (ROOT / "stop.sh").read_text()
        self.assertNotIn("coordinator and four", text)
        self.assertIn("configured Spark rank", text)

    def test_release_stop_services_covers_six_configured_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "stops"
            stub = f'''#!/usr/bin/env bash
printf '%s %s\\n' "$(basename "$0")" "$*" >> {log}
exit 0
'''
            for name in ("docker", "ssh"):
                path = root / name
                path.write_text(stub)
                path.chmod(0o755)
            setup = f'''set -euo pipefail
source scripts/release-common.sh
SPARK_COUNT=6
SPARK_0_HOST=h0
SPARK_1_HOST=h1
SPARK_2_HOST=h2
SPARK_3_HOST=h3
SPARK_4_HOST=h4
SPARK_5_HOST=h5
EXPERT_PORT=19441
ADDR=127.0.0.1:18000
release_stop_services coord worker
'''
            result = subprocess.run(
                ["bash", "-euc", setup, "test"], cwd=ROOT, text=True, capture_output=True,
                env=dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"]),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log.read_text() if log.exists() else ""
            for host in ("h0", "h1", "h2", "h3", "h4", "h5"):
                self.assertIn(host, log_text)


class ManifestWriterTest(unittest.TestCase):
    """The built-role manifest must be derived from validated build outputs."""

    def role_export(self, root: Path, role: str, *, intermediate: int = 1152, schema: int = 1,
                    capability=(12, 1), capacities=(1, 16, 80, 256, 1024, 4096),
                    object_bytes: bytes = b"object-bytes",
                    declared_bytes: bytes | None = None) -> None:
        export = root / f"v41_spark_{role}_experts"
        export.mkdir(parents=True, exist_ok=True)
        object_name = f"v41_{role}_m1.o"
        (export / object_name).write_bytes(object_bytes)
        manifest = {
            "schema": schema,
            "role": f"spark_{role}",
            "spark_tp_degree": {"tp2": 2, "tp3": 3}[role],
            "capability": list(capability),
            "geometry": {
                "experts": 384,
                "hidden": 5120,
                "intermediate": intermediate,
                "kernel_intermediate": intermediate,
                "topk": 6,
            },
            "variants": [{"capacity_rows": capacity} for capacity in capacities],
            "artifact_sha256": {
                object_name: hashlib.sha256(
                    object_bytes if declared_bytes is None else declared_bytes
                ).hexdigest(),
            },
            "sparkinfer_revision": "test-revision",
        }
        (export / "v41_experts.json").write_text(json.dumps(manifest))

    def write(self, root: Path, requested: str, *, role: str = "expert",
              library: bool = True, output: str = "V41_EXPERT_TP_AOT.json") -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable, str(ROOT / "scripts/write-v41-expert-tp-manifest.py"),
            "--role", role, "--requested", requested,
            "--native-build-dir", str(root),
            "--output", str(root / output),
        ]
        if library:
            (root / "libds41rt_native.so").write_bytes(b"fake-library")
            command += ["--native-library", str(root / "libds41rt_native.so")]
        return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)

    def test_valid_role_manifest_binds_geometry_and_library(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.role_export(root, "tp2", intermediate=1152)
            result = self.write(root, "tp2")
            self.assertEqual(result.returncode, 0, result.stderr)
            document = json.loads((root / "V41_EXPERT_TP_AOT.json").read_text())
            self.assertEqual(document["spark_tp_roles"], ["tp2"])
            self.assertEqual(document["manifests"]["tp2"]["geometry"]["intermediate"], 1152)
            self.assertEqual(len(document["native_library_sha256"]), 64)
            self.assertIn(document["symbols_verified"], (True, False))

    def test_default_empty_role_manifest_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.write(root, "")
            self.assertEqual(result.returncode, 0, result.stderr)
            document = json.loads((root / "V41_EXPERT_TP_AOT.json").read_text())
            self.assertEqual(document["spark_tp_roles"], [])
            self.assertEqual(document["manifests"], {})

    def test_stale_or_wrong_exports_cannot_advertise_a_role(self) -> None:
        cases = (
            ("schema", dict(schema=2), "schema"),
            ("capability", dict(capability=(12, 0)), "capability"),
            ("tp2_geometry", dict(intermediate=576), "geometry.intermediate"),
            ("missing_capacity", dict(capacities=(1, 16, 80)), "capacities"),
            ("wrong_object_hash", dict(object_bytes=b"file-bytes", declared_bytes=b"declared"), "hashes to"),
        )
        for name, kwargs, message in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    self.role_export(root, "tp2", **kwargs)
                    result = self.write(root, "tp2")
                    self.assertEqual(result.returncode, 2, result.stdout)
                    if message:
                        self.assertIn(message, result.stderr)

    def test_role_requires_native_library_and_expert_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.role_export(root, "tp2", intermediate=1152)
            result = self.write(root, "tp2", library=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--native-library is required", result.stderr)
            result = self.write(root, "tp2", role="coordinator")
            self.assertEqual(result.returncode, 2)
            self.assertIn("only the expert role", result.stderr)

    def test_unknown_role_is_a_parse_error_with_status_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.write(root, "tp4")
            self.assertEqual(result.returncode, 2)
            self.assertIn("unsupported Spark TP role", result.stderr)


class BuildScopeTest(unittest.TestCase):
    """Dry-run build/distribution paths must cover six hosts without touching
    Docker, SSH or any image."""

    def stub_path(self, root: Path) -> tuple[dict, Path]:
        stub_bin = root / "bin"
        stub_bin.mkdir()
        marker = root / "called"
        for name in ("docker", "ssh", "rsync", "nvidia-smi"):
            path = stub_bin / name
            path.write_text(
                "#!/usr/bin/env bash\n"
                f'echo "{name} $*" >>"{marker}"\n'
                "exit 1\n"
            )
            path.chmod(0o755)
        env = dict(os.environ, PATH=f"{stub_bin}{os.pathsep}{os.environ['PATH']}")
        return env, marker

    def test_build_dry_run_six_host_with_explicit_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env, marker = self.stub_path(root)
            result = subprocess.run(
                ["./build.sh", "--config", str(EXAMPLES / "tp3ep2-native.config"), "--dry-run"],
                cwd=ROOT, text=True, capture_output=True, env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("build hosts (6)", result.stdout)
            self.assertIn("V41 Spark expert roles: tp3", result.stdout)
            self.assertFalse(marker.exists(), marker.read_text() if marker.exists() else "")

    def test_wip_dry_run_six_host_official_only_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env, marker = self.stub_path(root)
            env["DS41RT_WIP_EXL3_AOT"] = "OFF"
            env["DS41RT_WIP_NVFP4_AOT"] = "OFF"
            result = subprocess.run(
                ["./wip.sh", "--config", str(EXAMPLES / "tp2ep3-native.config"), "--dry-run"],
                cwd=ROOT, text=True, capture_output=True, env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Spark hosts (6)", result.stdout)
            self.assertIn("V41 Spark expert roles: tp2", result.stdout)
            self.assertIn("EXL3 AOT: OFF; NVFP4 AOT: OFF", result.stdout)
            self.assertFalse(marker.exists(), marker.read_text() if marker.exists() else "")

    def test_helpers_plumb_roles_and_official_only_scope(self) -> None:
        release_helper = (ROOT / "scripts/build-release-artifacts.sh").read_text()
        wip_helper = (ROOT / "scripts/build-wip-artifacts.sh").read_text()
        dockerfile = (ROOT / "docker/Dockerfile.release").read_text()
        self.assertIn('-DDS41RT_V41_SPARK_TP_ROLES="$spark_tp_roles"', release_helper)
        self.assertIn('-DDS41RT_V41_SPARK_TP_ROLES="$spark_tp_roles"', wip_helper)
        self.assertIn("exl3_aot=\"${DS41RT_WIP_EXL3_AOT:-ON}\"", wip_helper)
        self.assertIn('-DDS41RT_ENABLE_V41_EXL3_AOT="$exl3_aot"', wip_helper)
        self.assertIn('if [[ "$exl3_aot" == ON ]]; then', wip_helper)
        self.assertIn("io.ds41rt.v41.spark_tp_roles", dockerfile)
        self.assertIn("V41_EXPERT_TP_AOT.json", dockerfile)

    def test_run_sh_accepts_actual_placement_gpus_and_holds_single_rtx(self) -> None:
        release = (ROOT / "run.sh").read_text()
        self.assertIn('--argjson gpus "$RELEASE_RTX_GPUS"', release)
        self.assertIn(".rtx_gpus == $gpus", release)
        # Single-RTX placement remains held for the daemon boot-order refactor.
        self.assertIn("((RELEASE_RTX_GPUS != 2)) || placement_directory=/run/ds41rt-placement", release)

    def test_shared_config_has_no_topology_to_rtx_hardcode(self) -> None:
        common = (ROOT / "scripts/release-common.sh").read_text()
        self.assertNotIn("one-RTX layout", common)
        self.assertNotIn("two-RTX layout", common)

    def test_run_wip_rejects_replicated_wire_on_the_legacy_backend(self) -> None:
        launcher = (ROOT / "scripts/run-wip.sh").read_text()
        self.assertIn("does not implement the replicated", launcher)
        self.assertNotIn('"$SPARK_COUNT" == 4 || "$SPARK_COUNT" == 6', launcher)


CANDIDATE = ROOT / "scripts" / "run-tp-ep-native-candidate.sh"

CANDIDATE_STUB = r'''#!/usr/bin/env python3
import os, re, sys
name = os.path.basename(sys.argv[0]); args = sys.argv[1:]
scenario = os.environ.get("STUB_SCENARIO", "normal")
worker_status = os.environ.get("STUB_WORKER_STATUS", "running")
log_offset = os.environ.get("STUB_LOG_OFFSET", "0")
log_full = os.environ.get("STUB_LOG_FULL", "")
plan = os.environ.get("STUB_PLAN", "")
def status_for(joined):
    if "expert" in joined:
        print(f"{worker_status} 123")
    else:
        print("stopped" if scenario == "child-exit" else "running 123")
def tail_from(joined):
    match = re.search(r"\+(\d+)", joined)
    start = int(match.group(1)) if match else 1
    sys.stdout.write(log_full[start-1:])
if name == "ssh":
    if scenario == "ssh-fail":
        sys.exit(255)
    remote = " ".join(args)
    if "docker inspect" in remote:
        print("true")
    elif " stat -c" in remote:
        if os.environ.get("STUB_SSH_STAT_EXIT") == "1":
            sys.exit(255)
        print(log_offset)
    elif " tail -c" in remote:
        tail_from(remote)
    elif "plan.json" in remote and " cat " in remote:
        sys.stdout.write(plan)
    elif "wip-process.sh" in remote and " status " in remote:
        status_for(remote)
    sys.exit(0)
if name == "curl":
    # Health and model-list probes both succeed; the model id matches.
    print('{"object":"list","data":[{"id":"deepseek-ai/DeepSeek-V4.1-Flash"}]}')
    sys.exit(0)
if args[:1] == ["inspect"]:
    print("true"); sys.exit(0)
if args[:1] == ["exec"]:
    tokens = list(args[1:]); cmd = []; i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("-d", "-i", "-t"):
            i += 1; continue
        if t in ("-e", "--env", "-w", "--workdir"):
            i += 2; continue
        if t.startswith("-"):
            i += 1; continue
        cmd = tokens[i:]; break
    if not cmd:
        sys.exit(0)
    rest = cmd[1:]
    if rest[:1] == ["test"]:
        # `test -e <dir>` is the stale-placement probe (absent on a clean run);
        # `test -s <file>` is the per-role wip-process existence preflight.
        if rest[1:2] == ["-e"]:
            sys.exit(0 if scenario == "stale" else 1)
        sys.exit(0)
    if rest[:1] == ["cat"]:
        if scenario == "child-exit":
            sys.exit(1)
        sys.stdout.write(plan); sys.exit(0)
    if rest[:1] == ["stat"]:
        print(log_offset); sys.exit(0)
    if rest[:1] == ["sh"] or rest[:1] == ["bash"]:
        joined = " ".join(rest)
        if "stat -c" in joined:
            print(log_offset); sys.exit(0)
        sys.exit(0)
    if rest[:1] == ["tail"]:
        tail_from(" ".join(rest)); sys.exit(0)
    if rest and rest[0].endswith("wip-process.sh") and len(rest) >= 2:
        if rest[1] == "status":
            status_for(" ".join(rest))
        sys.exit(0)
    sys.exit(0)
sys.exit(0)
'''

COORDINATOR_CONTROL_FLAGS = (
    "--prefill-batch-tokens",
    "--concurrency",
    "--prefix-cache-entries",
    "--max-context-tokens",
    "--max-output-tokens",
    "--http-queue-depth",
    "--http-queue-wait-ms",
    "--dspark",
    "--host-cache-bytes",
)


OPS_GPU_UUIDS = (
    "GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0,"
    "GPU-95f8f212-9131-df99-fd53-7535965197d7"
)

# Exact shape of the real G1 worker log line
# (runs/tp-ep-preflight/g1-live/raw/dodo-expert.log): the tracing formatter
# inserts ANSI dim/italic codes between a field name and its `=`.
ANSI_READY_LINE = (
    "\x1b[2m2026-09-20T07:13:46.765350Z\x1b[0m \x1b[32m INFO\x1b[0m "
    "\x1b[2mds41rt::v41_experts::service::local\x1b[0m\x1b[2m:\x1b[0m "
    "native local RoCE expert worker ready "
    "\x1b[3mrank\x1b[0m\x1b[2m=\x1b[0m{rank} "
    "\x1b[3mworld\x1b[0m\x1b[2m=\x1b[0m4 "
    "\x1b[3mcapacity\x1b[0m\x1b[2m=\x1b[0m4096 "
    "\x1b[3mfirst_layer\x1b[0m\x1b[2m=\x1b[0m20 "
    "\x1b[3mlayers\x1b[0m\x1b[2m=\x1b[0m20\n"
)
ANSI_READY_LINES = "".join(ANSI_READY_LINE.format(rank=rank) for rank in range(4))


class CandidateLauncherTest(unittest.TestCase):
    def plan(self, config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(CANDIDATE), "plan", "--config", str(config), *extra],
            cwd=ROOT, text=True, capture_output=True,
        )

    def test_plan_forwards_every_matched_control_and_orders_the_handshake(self) -> None:
        result = self.plan(
            EXAMPLES / "tp2ep2-native.config",
            "--coordinator-cuda-visible-devices", OPS_GPU_UUIDS,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        coordinator = next(line for line in lines if line.startswith("CMD coordinator:"))
        for flag in COORDINATOR_CONTROL_FLAGS:
            self.assertIn(flag, coordinator, flag)
        self.assertIn("--spark-tp 2", coordinator)
        self.assertIn("--spark-ep 2", coordinator)
        self.assertIn("--placement-directory", coordinator)
        # DS41RT_NATIVE_LIB must be present before the container and match --native-lib.
        self.assertIn("-e DS41RT_NATIVE_LIB=/scratch/coord-native/libds41rt_native.so", coordinator)
        self.assertIn("-e DS41RT_WIP_RUNTIME_ROOT=/scratch/candidate/run", coordinator)
        self.assertIn("-e RUST_LOG=info", coordinator)
        # The plan render escapes the CSV comma; both UUIDs and the CSV order
        # still appear in the coordinator command.
        self.assertIn(
            "-e CUDA_VISIBLE_DEVICES=GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0",
            coordinator,
        )
        self.assertIn("GPU-95f8f212-9131-df99-fd53-7535965197d7", coordinator)
        self.assertLess(
            coordinator.index("DS41RT_NATIVE_LIB=/scratch/coord-native/libds41rt_native.so"),
            coordinator.index("ds41rt-tpep-nvme-dev"),
        )
        self.assertLess(
            coordinator.index("CUDA_VISIBLE_DEVICES="),
            coordinator.index("ds41rt-tpep-nvme-dev"),
        )
        self.assertIn(f"coordinator_cuda_visible_devices={OPS_GPU_UUIDS}", result.stdout)
        self.assertIn("--native-lib /scratch/coord-native/libds41rt_native.so", coordinator)
        self.assertIn("expert_capacity=4096", result.stdout)
        # Plan handshake before any worker, worker readiness before ack, gate last.
        order = [line for line in lines if line.startswith("STEP")]
        self.assertLess(order.index("STEP coordinator-start"), order.index("STEP expert-start rank=0"))
        self.assertLess(order.index("STEP expert-start rank=0"), order.index("STEP worker-ready"))
        self.assertLess(order.index("STEP worker-ready"), order.index("STEP plan-ack"))
        self.assertLess(order.index("STEP plan-ack"), order.index("STEP readiness-gate"))
        experts = [line for line in lines if line.startswith("CMD expert-")]
        self.assertEqual(len(experts), 4)
        for rank, line in enumerate(experts):
            self.assertIn(f"--rank {rank}", line)
            self.assertIn("--world 4", line)
            self.assertIn(f"--rank {rank} --world 4 --spark-tp 2 --spark-ep 2", line)
            self.assertIn("-e DS41RT_NATIVE_LIB=/scratch/candidate/libds41rt_native.so", line)
            self.assertIn("-e RUST_LOG=info", line)
            self.assertIn("--native-lib /scratch/candidate/libds41rt_native.so", line)
            self.assertLess(line.index("DS41RT_NATIVE_LIB="), line.index("ds41rt-tpep-dev"))
            # The GPU pin is coordinator-only.
            self.assertNotIn("CUDA_VISIBLE_DEVICES", line)

    def test_coordinator_gpu_pin_is_validated(self) -> None:
        config = EXAMPLES / "tp2ep2-native.config"
        u1 = "GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0"
        for pin, message in (
            (f"{u1},{u1}", "duplicate"),
            (u1, "exactly 2 UUID"),
            ("0,1", "not a physical GPU UUID"),
        ):
            with self.subTest(pin=pin):
                result = self.plan(config, "--coordinator-cuda-visible-devices", pin)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)

    def test_qualified_requires_the_coordinator_gpu_pin(self) -> None:
        digest = "a" * 64
        common = (
            "--qualified",
            "--expect-coordinator-lib-sha256", digest,
            "--expect-spark-lib-sha256", digest,
            "--expect-coordinator-daemon-sha256", digest,
            "--expect-spark-daemon-sha256", digest,
        )
        result = self.plan(EXAMPLES / "tp2ep2-native.config", *common)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--qualified requires --coordinator-cuda-visible-devices", result.stderr)
        result = self.plan(
            EXAMPLES / "tp2ep2-native.config", *common,
            "--coordinator-cuda-visible-devices", OPS_GPU_UUIDS,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_plan_six_rank_single_rtx_loads_from_layer_zero(self) -> None:
        result = self.plan(EXAMPLES / "tp3ep2-native.config")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertIn("topology tp=3 ep=2 world=6", lines)
        self.assertIn("first_layer=0", lines)
        self.assertNotIn("STEP plan-read", lines)
        experts = [line for line in lines if line.startswith("CMD expert-")]
        self.assertEqual(len(experts), 6)
        self.assertIn("rhea", [line for line in experts if "--rank 4" in line][0])
        self.assertIn("moa", [line for line in experts if "--rank 5" in line][0])
        self.assertTrue(all("--world 6 --spark-tp 3 --spark-ep 2" in line for line in experts))
        order = [line for line in lines if line.startswith("STEP")]
        self.assertLess(order.index("STEP expert-start rank=5"), order.index("STEP worker-ready"))
        self.assertLess(order.index("STEP worker-ready"), order.index("STEP coordinator-start"))

    def test_plan_supports_the_tp4_reference_arm(self) -> None:
        result = self.plan(EXAMPLES / "tp4ep1-explicit-native.config")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("topology tp=4 ep=1 world=4", result.stdout.splitlines())
        # No tp4 extra role: the reference arm runs without a role manifest.
        provider = next(line for line in result.stdout.splitlines() if line.startswith("CMD expert-0:"))
        self.assertIn("--world 4 --spark-tp 4 --spark-ep 1", provider)

    def test_default_config_without_explicit_topology_is_rejected(self) -> None:
        result = self.plan(CONFIG)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires explicit SPARK_TP", result.stderr)

    def test_examples_never_use_the_stale_secondary_rail(self) -> None:
        self.assertIn("SPARK_0_LANE_B=10.55.0.5", CONFIG.read_text())  # default unchanged
        for name in ("tp2ep2-native.config", "tp4ep1-explicit-native.config"):
            text = (EXAMPLES / name).read_text()
            self.assertNotIn("SPARK_0_LANE_B=10.55.0.5", text)
            self.assertNotIn("SPARK_1_LANE_B=10.55.0.6", text)
            self.assertIn("SPARK_0_LANE_B=10.55.0.7", text)
            self.assertIn("SPARK_1_LANE_B=10.55.0.8", text)
            self.assertIn("SPARK_2_LANE_B=10.55.0.9", text)
            self.assertIn("SPARK_3_LANE_B=10.55.0.10", text)

    def candidate(self, root: Path, *args: str, scenario: str = "normal",
                  env_extra: dict | None = None) -> subprocess.CompletedProcess[str]:
        stub_bin = root / "bin"
        stub_bin.mkdir(exist_ok=True)
        for name in ("docker", "ssh", "curl"):
            path = stub_bin / name
            path.write_text(CANDIDATE_STUB)
            path.chmod(0o755)
        env = dict(
            os.environ,
            PATH=f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
            STUB_SCENARIO=scenario,
            DS41RT_TPEP_L3_GRANT="1",
        )
        if env_extra:
            env.update(env_extra)
        return subprocess.run([str(CANDIDATE), *args], cwd=ROOT, text=True, capture_output=True, env=env)

    READY_PLAN = '{"version":1,"rtx_gpus":2,"nonce":"n","rtx_expert_layers":20,"spark_first_layer":20}'
    READY_PREFIX = "stale-prefix-line\n"

    def test_start_waits_for_worker_ready_before_reporting_ready(self) -> None:
        # Nonzero offset before the real ANSI line: the launcher must search only
        # the current epoch and strip ANSI before matching rank/world/first_layer.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.candidate(
                root, "start", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                "--host-artifact-root", str(root), "--run-id", "ready-run",
                env_extra={"STUB_PLAN": self.READY_PLAN,
                           "STUB_LOG_OFFSET": str(len(self.READY_PREFIX)),
                           "STUB_LOG_FULL": self.READY_PREFIX + ANSI_READY_LINES},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("worker ready", result.stdout)
            self.assertIn("candidate ready", result.stdout)
            self.assertLess(result.stdout.index("worker ready"), result.stdout.index("candidate ready"))

    def test_stale_worker_log_before_offset_does_not_satisfy_ready(self) -> None:
        # The readiness bytes precede the pre-start offset, so the current-epoch
        # slice is empty and the wait must time out rather than accept stale text.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            offset = str(len(self.READY_PREFIX))
            result = self.candidate(
                root, "start", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                "--host-artifact-root", str(root), "--run-id", "stale-log-run",
                env_extra={"STUB_PLAN": self.READY_PLAN, "STUB_LOG_OFFSET": offset,
                           "STUB_LOG_FULL": self.READY_PREFIX,
                           "DS41RT_TPEP_WORKER_READY_TIMEOUT_SECONDS": "1"},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("worker readiness timed out", result.stderr)

    def test_wrong_rank_and_split_line_do_not_satisfy_ready(self) -> None:
        wrong_rank = ANSI_READY_LINE.format(rank=10)  # rank=10 must not match rank=1
        split_line = (
            "\x1b[2m ds41rt: native local RoCE expert worker ready\x1b[0m\n"
            "\x1b[3mrank\x1b[0m\x1b[2m=\x1b[0m1 \x1b[3mworld\x1b[0m\x1b[2m=\x1b[0m4 "
            "\x1b[3mfirst_layer\x1b[0m\x1b[2m=\x1b[0m20\n"
        )
        for name, body in (("wrong-rank", wrong_rank), ("split-line", split_line)):
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    result = self.candidate(
                        root, "start", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                        "--host-artifact-root", str(root), "--run-id", f"{name}-run",
                        env_extra={"STUB_PLAN": self.READY_PLAN,
                                   "STUB_LOG_OFFSET": str(len(self.READY_PREFIX)),
                                   "STUB_LOG_FULL": self.READY_PREFIX + body,
                                   "DS41RT_TPEP_WORKER_READY_TIMEOUT_SECONDS": "1"},
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("worker readiness timed out", result.stderr)

    def test_ssh_stat_failure_is_fatal_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.candidate(
                root, "start", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                "--host-artifact-root", str(root), "--run-id", "stat-fail-run",
                env_extra={"STUB_PLAN": self.READY_PLAN, "STUB_SSH_STAT_EXIT": "1"},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("failed to read worker log size", result.stderr)

    def test_worker_death_before_ready_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.candidate(
                root, "start", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                "--host-artifact-root", str(root), "--run-id", "dead-worker-run",
                env_extra={"STUB_PLAN": self.READY_PLAN, "STUB_LOG_OFFSET": "0",
                           "STUB_LOG_FULL": "", "STUB_WORKER_STATUS": "stopped"},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exited before reporting ready", result.stderr)

    def test_stale_placement_directory_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.candidate(
                root, "start", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                "--host-artifact-root", str(root), "--run-id", "stale-run", scenario="stale",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("placement directory already exists", result.stderr)

    def test_coordinator_child_exit_fails_before_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.candidate(
                root, "start", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                "--host-artifact-root", str(root), "--run-id", "child-run", scenario="child-exit",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exited before publishing placement", result.stderr)

    def test_stop_reports_unreachable_hosts_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.candidate(
                root, "stop", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                "--host-artifact-root", str(root), scenario="ssh-fail",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unreachable", result.stderr)
            self.assertIn("had one or more failures", result.stderr)


if __name__ == "__main__":
    unittest.main()
