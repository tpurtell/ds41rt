"""Keep the unused FP8 producer out of NVFP4 router graphs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_nvfp4_router_skips_fp8_producer():
    source = (ROOT / "rust/crates/ds41rt-daemon/src/v41_backbone_router.rs").read_text()
    enqueue = source.split("unsafe fn enqueue(", 1)[1].split("pub unsafe fn execute(", 1)[0]
    assert "if !self.weights.nvfp4 {\n                self.input_quantizer" in enqueue
    assert enqueue.count("self.input_quantizer") == 1
    assert "self.b(if self.weights.nvfp4 { 0 } else { 5 })" in enqueue


def test_router_rebind_rejects_quantization_format_change():
    source = (ROOT / "rust/crates/ds41rt-daemon/src/v41_backbone_router.rs").read_text()
    assert "self.weights.nvfp4 == weights.nvfp4" in source
