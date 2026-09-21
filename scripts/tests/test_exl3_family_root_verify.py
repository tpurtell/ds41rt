#!/usr/bin/env python3
"""CPU-only regressions for verifying an EXL3 root with nested families.

Release images keep one well-known `exl3/` root. Multi-family builds nest
`exl3-kXX` packages beneath it, while a direct `manifest.json` in the root is
the legacy single-family layout. `build.sh` verifies and checksums the exported
root, so both layouts must be handled, strictly:

* every `exl3-*` family must be verified with the unchanged strict
  single-package validation -- one valid package must never mask a broken
  sibling, and a family without a manifest is an error rather than skipped;
* a root mixing a flat manifest with family packages is rejected as ambiguous;
* a root with no package at all is an error.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "python" / "tools" / "package_v41_exl3_aot.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("package_v41_exl3_aot", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(path: Path, role: str = "coordinator") -> None:
    path.write_text(json.dumps({"schema": "ds41rt.exl3-package.v1", "role": role}))


@pytest.fixture()
def recorded(monkeypatch):
    """Replace the strict single-package verifier with a recorder."""
    tool = load_tool()
    calls: list[Path] = []

    def fake_verify(package, revision=None, runtime=None, role=None):
        calls.append(Path(package))
        return {"role": "coordinator", "variants": [1]}

    monkeypatch.setattr(tool, "verify", fake_verify)
    return tool, calls


def test_legacy_flat_root_verifies_the_root_itself(tmp_path, recorded):
    tool, calls = recorded
    _manifest(tmp_path / "manifest.json")
    manifests = tool.verify_root(tmp_path)
    assert calls == [tmp_path]
    assert len(manifests) == 1


def test_nested_family_root_verifies_every_family(tmp_path, recorded):
    tool, calls = recorded
    for family in ("exl3-k23", "exl3-k34"):
        child = tmp_path / family
        child.mkdir()
        _manifest(child / "manifest.json")
    manifests = tool.verify_root(tmp_path)
    assert calls == [tmp_path / "exl3-k23", tmp_path / "exl3-k34"]
    assert len(manifests) == 2


def test_missing_family_manifest_fails_closed(tmp_path, recorded):
    """A broken sibling must fail the whole root, not be skipped."""
    tool, calls = recorded
    good = tmp_path / "exl3-k23"
    good.mkdir()
    _manifest(good / "manifest.json")
    (tmp_path / "exl3-k34").mkdir()  # no manifest.json
    with pytest.raises(ValueError, match="missing its manifest.json"):
        tool.verify_root(tmp_path)


def test_ambiguous_flat_and_nested_root_is_rejected(tmp_path, recorded):
    tool, _calls = recorded
    _manifest(tmp_path / "manifest.json")
    child = tmp_path / "exl3-k34"
    child.mkdir()
    _manifest(child / "manifest.json")
    with pytest.raises(ValueError, match="ambiguous EXL3 root"):
        tool.verify_root(tmp_path)


def test_unexpected_package_directory_is_rejected(tmp_path, recorded):
    tool, _calls = recorded
    child = tmp_path / "k34-not-a-family"
    child.mkdir()
    _manifest(child / "manifest.json")
    with pytest.raises(ValueError, match="unexpected EXL3 package directory"):
        tool.verify_root(tmp_path)


def test_root_without_any_package_is_an_error(tmp_path, recorded):
    tool, _calls = recorded
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no EXL3 package or family manifest"):
        tool.verify_root(tmp_path)


def test_missing_root_is_an_error(tmp_path, recorded):
    tool, _calls = recorded
    with pytest.raises(ValueError, match="no EXL3 package or family manifest"):
        tool.verify_root(tmp_path / "absent")


def test_build_sh_verifies_the_exported_root():
    text = (REPO / "build.sh").read_text()
    assert 'verify --package "$repo_root/dist/$role/exl3"' in text


def test_build_sh_checksums_real_exl3_manifest_paths():
    """The dist checksum list must not name a flat manifest that never exists."""
    text = (REPO / "build.sh").read_text()
    checksum_block = text.split("sha256sum \\", 1)[1].split(">SHA256SUMS", 1)[0]
    assert "coordinator/exl3/manifest.json" not in checksum_block
    assert "spark-expert/exl3/manifest.json" not in checksum_block
    assert '"${dist_exl3_manifests[@]}"' in checksum_block


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
