"""Read-only diagnostic candidates -> standard indexed shards -> V4.1 reload."""
import argparse
import json
from pathlib import Path
import sqlite3
import tempfile

import torch

from gptqmodel.utils.v41_checkpoint import _load_state
from gptqmodel.utils.v41_source import V41Source
from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import reconstruct_exl3_tensors
from export_inventory import packed_entries
from stream_shard import repack


def run(fixture):
    with sqlite3.connect(f"file:{fixture.resolve()}/run.sqlite?mode=ro", uri=True) as db:
        identity = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
        cases = []
        for projection in ("w1", "w3", "w2"):
            phase = "down" if projection == "w2" else "gate_up"
            for bits in (3, 4):
                row = db.execute("SELECT key, record FROM artifacts WHERE key LIKE ? ORDER BY key LIMIT 1",
                                 (f"blocks/mtp/000/{phase}/expert-%/{projection}-k{bits}",)).fetchone()
                if row is None:
                    raise ValueError("diagnostic fixture is missing a K3/K4 projection")
                key, encoded = row
                record = json.loads(encoded)
                provenance = {**identity, "artifact": key}
                path = fixture / record["path"]
                state = _load_state(path, expected_sha256=record["sha256"], expected_provenance=provenance, kind="projection")
                entries = packed_entries(path, provenance=provenance, bits=bits, projection=projection)
                name = f"mtp.0.ffn.experts.0.{projection}"
                with tempfile.TemporaryDirectory(prefix="ds41rt-export-reload-") as directory:
                    root = Path(directory)
                    filename = "model-00001-of-00001.safetensors"
                    repack({name + "." + suffix: (entry["path"], entry["tensor"]) for suffix, entry in entries.items()}, root / filename)
                    (root / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
                    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                        name + "." + suffix: filename for suffix in entries}}))
                    source = V41Source(root)
                    loaded = source.packed_projection(name)
                    for suffix in entries:
                        torch.testing.assert_close(loaded[suffix], state["packed"][suffix], rtol=0, atol=0)
                    expected = reconstruct_exl3_tensors(state["packed"], device="cuda:0", dtype=torch.bfloat16).T.contiguous()
                    actual = source.decoded(name + ".weight", "cuda:0")
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    cases.append(dict(projection=projection, bits=bits, shape=list(actual.shape), exact=True))
                    del actual, expected, loaded
                del state
    print(json.dumps(dict(status="passed", scope="six actual candidate export/reload cases, not full-model validation", cases=cases)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    run(parser.parse_args().fixture)
