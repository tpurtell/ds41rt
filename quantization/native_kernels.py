"""Serialize TileLang/TVM host registry access, not GPU execution.

The pinned free-threaded runtime can race imported-module lookup when two RTX
threads create/dispatch JIT kernels simultaneously. Keep the whole host dispatch
under one reentrant lock; CUDA launches remain asynchronous on each device.
"""
from functools import wraps
import threading


_dispatch_lock = threading.RLock()


def _guard(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        with _dispatch_lock:
            result = function(*args, **kwargs)
            # JIT factories return a callable kernel whose first invocation can
            # lazily touch the same TVM module registry as compilation.
            return _guard(result) if callable(result) else result
    return guarded


class SerializedHostKernels:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        value = getattr(self.module, name)
        return _guard(value) if callable(value) else value
