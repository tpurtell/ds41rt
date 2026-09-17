from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SELECTOR = ROOT / "scripts/select-release-gpus.py"
PRIMARY = "GPU-00000000-0000-0000-0000-000000000000"
SECONDARY = "GPU-11111111-1111-1111-1111-111111111111"


class ReleaseGpuSelectionTest(unittest.TestCase):
    def test_negotiated_boundary_allows_pair_below_twenty_layer_budget(self):
        ordinary = self.invoke(primary_free=65000, secondary_free=65000)
        self.assertEqual(ordinary.returncode, 0, ordinary.stderr)
        self.assertEqual(json.loads(ordinary.stdout)['count'], 1)
        negotiated = self.invoke(primary_free=65000, secondary_free=65000,
                                 extra=('--minimum-expert-layers', '1'))
        self.assertEqual(negotiated.returncode, 0, negotiated.stderr)
        self.assertEqual(json.loads(negotiated.stdout)['count'], 2)
        too_small = self.invoke(mode='2', primary_free=15000, secondary_free=15000,
                                extra=('--minimum-expert-layers', '1'))
        self.assertNotEqual(too_small.returncode, 0)

    def invoke(
        self,
        *,
        mode: str = "auto",
        primary_free: int = 97_000,
        secondary_free: int = 96_900,
        p2p: bool = True,
        extra: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            peer_status = "OK" if p2p else "NS"
            fake = Path(directory) / "nvidia-smi"
            fake.write_text(
                "#!/usr/bin/env bash\n"
                "case \"$1\" in\n"
                "  --query-gpu=*)\n"
                f"    echo '0, {PRIMARY}, 00000000:11:00.0, 97887, {primary_free}'\n"
                f"    echo '1, {SECONDARY}, 00000000:E1:00.0, 97887, {secondary_free}'\n"
                "    ;;\n"
                "  --query-compute-apps=*)\n"
                f"    echo '{SECONDARY}, 42, 96500'\n"
                f"    echo '{SECONDARY}, 99, 90000'\n"
                "    ;;\n"
                "  topo)\n"
                "    [[ \"$2 $3\" == '-p2p r' ]] || exit 9\n"
                f"    printf 'GPU0 GPU1\\nGPU0 X {peer_status}\\nGPU1 {peer_status} X\\n'\n"
                "    ;;\n"
                "  *) exit 9 ;;\n"
                "esac\n"
            )
            fake.chmod(0o755)
            return subprocess.run(
                [
                    "python3",
                    str(SELECTOR),
                    "--mode",
                    mode,
                    "--primary-uuid",
                    PRIMARY,
                    "--concurrency",
                    "16",
                    "--max-context-tokens",
                    "1048576",
                    "--retained-turns",
                    "24",
                    "--nvidia-smi",
                    str(fake),
                    *extra,
                ],
                text=True,
                capture_output=True,
                env=os.environ.copy(),
            )

    def test_auto_selects_ordered_pair_for_default_pool(self) -> None:
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        selected = json.loads(result.stdout)
        self.assertEqual(selected["count"], 2)
        self.assertEqual(selected["decision"], "automatic-dual")
        self.assertEqual(selected["groups"], 28_736)
        self.assertEqual(selected["required_mib"], [96_176, 96_778])
        self.assertEqual([gpu["uuid"] for gpu in selected["gpus"]], [PRIMARY, SECONDARY])

    def test_auto_falls_back_without_memory_or_bidirectional_peer_reads(self) -> None:
        for free, p2p in ((96_000, True), (96_900, False)):
            with self.subTest(free=free, p2p=p2p):
                result = self.invoke(secondary_free=free, p2p=p2p)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["count"], 1)

    def test_force_two_reports_memory_requirements(self) -> None:
        result = self.invoke(mode="2", secondary_free=96_000)
        self.assertEqual(result.returncode, 2)
        self.assertIn("forced two-RTX mode is infeasible", result.stderr)
        self.assertIn("96176/96778 MiB", result.stderr)

    def test_force_one_does_not_apply_dual_pool_constraints(self) -> None:
        result = self.invoke(mode="1", extra=("--kv-pool-size", "1B"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["count"], 1)
        invalid = self.invoke(mode="1", extra=("--memory-reservation", "101%"))
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("exceeds device total", invalid.stderr)

    def test_only_named_replacement_process_memory_is_reclaimable(self) -> None:
        without_reclaim = self.invoke(secondary_free=1_000)
        self.assertEqual(json.loads(without_reclaim.stdout)["count"], 1)
        with_reclaim = self.invoke(
            secondary_free=1_000, extra=("--reclaim-pid", "42")
        )
        selected = json.loads(with_reclaim.stdout)
        self.assertEqual(selected["count"], 2)
        self.assertEqual(selected["gpus"][1]["reclaim_mib"], 96_500)
        self.assertEqual(selected["gpus"][1]["effective_free_mib"], 97_500)

    def test_explicit_small_pool_changes_feasibility_but_reservation_is_a_ceiling(self) -> None:
        explicit = self.invoke(
            primary_free=94_000,
            secondary_free=94_000,
            extra=("--kv-pool-size", "1GiB"),
        )
        self.assertEqual(json.loads(explicit.stdout)["count"], 2)
        reservation = self.invoke(extra=("--memory-reservation", "90%"))
        self.assertEqual(json.loads(reservation.stdout)["count"], 1)


if __name__ == "__main__":
    unittest.main()
