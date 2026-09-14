"""Continuously refilled, bounded search; journal access stays on its owner."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading

from gptqmodel.utils.v41_checkpoint import _load_state
from gptqmodel.utils.exl3_inline_mixed import projection_score


class CandidateQueue:
    def __init__(self, driver):
        self.driver = driver
        self.workers = getattr(driver.search, "max_workers", 1)
        if type(self.workers) is not int or not 1 <= self.workers <= 16:
            raise ValueError("invalid search backend concurrency")
        # Queued jobs hold paths/metadata, not Hessian tensors. At most workers
        # load Hessians; completed results plus queued work stay within 2*workers.
        self.limit = 2 * self.workers
        self.pending = {}
        self.high_water = 0
        self.stopped = threading.Event()
        self.pool = None

    def __enter__(self):
        if self.workers > 1:
            self.pool = ThreadPoolExecutor(max_workers=self.workers)
        return self

    def _search(self, name, bits, record, provenance):
        try:
            if self.stopped.is_set():
                raise RuntimeError("search queue stopped after failure")
            captured = _load_state(self.driver.journal.root / record["path"],
                expected_sha256=record["sha256"], expected_provenance=provenance, kind="hessian")
            if self.stopped.is_set():
                raise RuntimeError("search queue stopped after failure")
            packed, metrics = self.driver.search(name, captured["hessian"], bits)
            return dict(packed=packed, quantizer_metrics=metrics, route_evidence=captured["evidence"])
        except BaseException:
            self.stopped.set()
            raise

    def harvest(self, *, block=False):
        if block and self.pending:
            wait(self.pending, return_when=FIRST_COMPLETED)
        for future in tuple(self.pending):
            if not future.done():
                continue
            key, bits, hessian_key = self.pending.pop(future)
            result = future.result()
            projection_score(result)
            self.driver._publish(key, "projection", result, (hessian_key,))
            self.driver.progress(dict(event="candidate_committed", key=key, bits=bits))
        if self.stopped.is_set():
            # A worker can signal just before its future becomes done. Drain
            # that failure rather than submitting more work in this small gap.
            if self.pending:
                wait(self.pending)
                self.harvest()
            raise RuntimeError("search queue stopped after failure")

    def submit(self, job):
        if self.workers == 1:
            self.driver._candidate(*job)
            return
        self.harvest()
        while len(self.pending) >= self.limit:
            self.harvest(block=True)
        prefix, expert, projection, bits, hessian_key, source_prefix = job
        key = f"{prefix}/expert-{expert:03d}/{projection}-k{bits}"
        if self.driver._load(key, "projection") is not None:
            return
        record = self.driver.journal.get(hessian_key, verify=False)
        if record is None or record["kind"] != "hessian":
            raise ValueError("search requires a committed Hessian")
        future = self.pool.submit(self._search, f"{source_prefix}.ffn.experts.{expert}.{projection}", bits,
                                  record, {**self.driver.identity, "artifact": hessian_key})
        self.pending[future] = (key, bits, hessian_key)
        self.high_water = max(self.high_water, len(self.pending))

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                while self.pending:
                    self.harvest(block=True)
        finally:
            self.stopped.set()
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)
            self.pending.clear()
