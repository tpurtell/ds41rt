from collections import Counter
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block_driver import BlockDriver
from candidate_queue import CandidateQueue
from run_store import RunStore


class CandidateQueueTest(unittest.TestCase):
    def driver(self, root, search, count):
        identity = {"test": "continuous-candidates"}
        journal = RunStore(root, identity)
        self.addCleanup(journal.close)
        driver = BlockDriver(None, journal, identity, device="cpu", search=search)
        jobs = []
        for expert in range(count):
            key = f"phase/expert-{expert:03d}/hessian"
            driver._publish(key, "hessian", dict(hessian={"H": torch.eye(2), "count": 2},
                evidence={"expert_gate_squared_mass_fraction": 1.0}), ())
            jobs.append(("phase", expert, "w1", 3, key, "layers.0"))
        owner = threading.get_ident()
        original_get, original_publish = journal.get, driver._publish
        def get(*args, **kwargs):
            self.assertEqual(threading.get_ident(), owner)
            return original_get(*args, **kwargs)
        def publish(*args, **kwargs):
            self.assertEqual(threading.get_ident(), owner)
            return original_publish(*args, **kwargs)
        journal.get, driver._publish = get, publish
        return driver, jobs

    @staticmethod
    def result(bits):
        return {"bits": torch.tensor(bits)}, {"hessian_weighted_relative_error": 1.0}

    def test_fast_worker_advances_past_slow_first_job(self):
        reached = threading.Event()
        started = []
        def search(name, hessian, bits):
            expert = int(name.split(".")[-2])
            started.append(expert)
            if expert == 0:
                if not reached.wait(5):
                    raise AssertionError("fast worker stranded behind slow batch")
            if expert == 7:
                reached.set()
            return self.result(bits)
        search.max_workers = 2
        with tempfile.TemporaryDirectory() as root:
            driver, jobs = self.driver(root, search, 12)
            with CandidateQueue(driver) as queue:
                # Two conceptual capture subsets, deliberately without a drain.
                for job in jobs[:4]:
                    queue.submit(job)
                for job in jobs[4:]:
                    queue.submit(job)
            self.assertTrue(reached.is_set())
            self.assertLessEqual(queue.high_water, 4)
            self.assertEqual(sorted(started), list(range(12)))
            driver._candidates(jobs)
            self.assertEqual(len(started), 12)  # committed candidates reused

    def test_heterogeneous_workers_take_work_by_availability_not_fixed_ratio(self):
        for ratio in (4, 7):
            with self.subTest(ratio=ratio), tempfile.TemporaryDirectory() as root:
                condition = threading.Condition()
                busy = set()
                counts = Counter()
                def search(name, hessian, bits):
                    with condition:
                        condition.wait_for(lambda: len(busy) < 6)
                        slot = next(i for i in range(6) if i not in busy)
                        busy.add(slot)
                        counts[slot] += 1
                    try:
                        # Uneven individual durations as well as device speeds.
                        expert = int(name.split(".")[-2])
                        time.sleep(.01 * (1 if slot < 2 else ratio) * (1 + .1 * (expert % 3)))
                        return self.result(bits)
                    finally:
                        with condition:
                            busy.remove(slot)
                            condition.notify_all()
                search.max_workers = 6
                driver, jobs = self.driver(root, search, 96)
                with CandidateQueue(driver) as queue:
                    for job in jobs:
                        queue.submit(job)
                self.assertEqual(set(counts), set(range(6)))
                self.assertGreater((counts[0] + counts[1]) / 2, max(counts[i] for i in range(2, 6)) * 2)
                self.assertEqual(sum(counts.values()), 96)
                self.assertLessEqual(queue.high_water, 12)

    def test_failure_stops_dispatch_and_does_not_publish_failed_candidate(self):
        def search(name, hessian, bits):
            raise RuntimeError("injected search failure")
        search.max_workers = 2
        with tempfile.TemporaryDirectory() as root:
            driver, jobs = self.driver(root, search, 12)
            with self.assertRaisesRegex(RuntimeError, "failure"):
                driver._candidates(jobs)
            count = driver.journal.db.execute("SELECT count(*) FROM artifacts WHERE key LIKE '%/w1-k3'").fetchone()[0]
            self.assertEqual(count, 0)
