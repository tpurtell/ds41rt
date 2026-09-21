#!/usr/bin/env python3
"""CPU-only regressions for the universal release image pair.

One published pair has to serve every approved native topology: the ARM64 (SM121)
Spark expert image carries the historical TP4 shard plus the TP2, TP3 and TP6
replicated-group shards, and the x86_64 coordinator image carries none of them
because Spark expert roles are expert-only. `./run.sh` selects the mode from
`SPARK_TP`/`SPARK_EP`, so a published image must never advertise a role it cannot
honor, and a build must not label a partial export as the resolved request.

No Docker, SSH, GPU, Cargo or CMake is touched. Each test drives the real producer
text: the `build.sh` canonicalizer and post-build comparison are extracted between
their markers and executed in bash; the `push-containers.sh` guard is extracted and
executed, with its call site checked against the tag/push order; and the
`docker/Dockerfile.release` bake-time corroboration is checked both as Dockerfile
structure (a multi-line shell block that loses a continuation becomes unparsable
instructions) and as the real `jq` filter it runs, against coordinator, universal,
subset and mismatched manifests.

`scripts/write-v41-expert-tp-manifest.py` stays the only writer of the built-role
manifest: it derives every entry from the AOT export directory and the library that
was linked, never from a caller-supplied claim. The Dockerfile check below is what
proves the library that ships is still the one that was hashed.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "build.sh"
PUSH = REPO / "push-containers.sh"
DOCKERFILE = REPO / "docker" / "Dockerfile.release"
RELEASE_COMMON = REPO / "scripts" / "release-common.sh"

ROLES_BLOCK = ("# release-spark-tp-roles:start", "# release-spark-tp-roles:end")
POSTCHECK_BLOCK = ("# release-spark-tp-roles-postcheck:start",
                   "# release-spark-tp-roles-postcheck:end")
GUARD_BLOCK = ("# push-universal-role-guard:start", "# push-universal-role-guard:end")

UNIVERSAL = "tp2;tp3;tp6"


def _between(path: Path, markers: tuple[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    start, end = markers
    for marker in markers:
        assert marker in text, f"{path.name} lost its {marker!r} marker"
    return text.split(start, 1)[1].split(end, 1)[0]


def _bash(script: str, positional: list[str] | None = None):
    return subprocess.run(
        ["bash", "-c", script, "bash", *(positional or [])],
        capture_output=True, text=True, timeout=60,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin"}, check=False,
    )


# ---------------------------------------------------------------------------
# build.sh: one canonicalizer, shared by the selection and the post-build check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spark_tp,spark_ep", [
    ("", ""), ("4", "1"), ("2", "2"), ("2", "3"), ("3", "2"), ("6", "1"),
])
def test_universal_roles_are_the_default_for_every_configuration(spark_tp, spark_ep):
    """The resolved role set no longer depends on the configured topology.

    A topology-derived role set meant `./build.sh` on the shipped default
    configuration published an image that `./run.sh` then refused for every
    explicit SPARK_TP/SPARK_EP configuration.
    """
    result = _bash(f"""
set -euo pipefail
source "{RELEASE_COMMON}"
SPARK_COUNT=4
SPARK_TP="{spark_tp}"
SPARK_EP="{spark_ep}"
{_between(BUILD, ROLES_BLOCK)}
printf 'ROLES=%s\\n' "$spark_tp_roles"
""")
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().splitlines()[-1] == f"ROLES={UNIVERSAL}", result.stdout


@pytest.mark.parametrize("requested,expected", [
    (UNIVERSAL, UNIVERSAL),
    ("tp6;tp2;tp3", UNIVERSAL),          # a permutation is the same set
    ("tp6", "tp6"),                      # bounded single-topology A/B
    ("tp2;tp6", "tp2;tp6"),              # documented non-contiguous subset
    ("", ""),                            # explicit legacy TP4-only rebuild
])
def test_subset_escape_hatch_stays_available(requested, expected):
    result = _bash(f"""
set -euo pipefail
source "{RELEASE_COMMON}"
export DS41RT_RELEASE_SPARK_TP_ROLES="{requested}"
{_between(BUILD, ROLES_BLOCK)}
printf 'ROLES=%s\\n' "$spark_tp_roles"
""")
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().splitlines()[-1] == f"ROLES={expected}", result.stdout
    if expected == UNIVERSAL:
        assert "NON-UNIVERSAL" not in result.stdout, result.stdout
    else:
        assert "NON-UNIVERSAL" in result.stdout, result.stdout


@pytest.mark.parametrize("raw,fragment", [
    ("tp5", "accepts only tp2, tp3 and tp6"),
    ("tp2;tp5", "accepts only tp2, tp3 and tp6"),
    ("spark", "accepts only tp2, tp3 and tp6"),
    ("tp2 tp3", "accepts only tp2, tp3 and tp6"),
    ("tp2;;tp3", "is not a ';'-separated role list"),   # no silent narrowing
    ("tp2;", "is not a ';'-separated role list"),
    (";tp2", "is not a ';'-separated role list"),
    ("tp2\n", "is not a ';'-separated role list"),
    ("tp2;tp2", "lists tp2 more than once"),
    (UNIVERSAL + ";tp2", "lists tp2 more than once"),
])
def test_malformed_role_lists_fail_closed(raw, fragment):
    result = _bash(f"""
set -euo pipefail
source "{RELEASE_COMMON}"
{_between(BUILD, ROLES_BLOCK)}
release_spark_tp_roles_canonical "$1" DS41RT_RELEASE_SPARK_TP_ROLES
""", positional=[raw])
    assert result.returncode == 2, (raw, result.stdout, result.stderr)
    assert fragment in result.stderr, (raw, result.stderr)


def test_role_allowlist_and_comparison_each_exist_once():
    text = BUILD.read_text(encoding="utf-8")
    assert text.count("release_spark_tp_roles_canonical() {") == 1
    # A second inline `case` would be a second allowlist free to drift.
    assert len(re.findall(r"tp2\|tp3\|tp6\)", text)) == 1
    assert '*";$spark_tp_roles;"*' not in text, "substring role comparison is back"


def test_postcheck_accepts_a_permutation_and_rejects_a_different_set():
    harness = f"""
set -euo pipefail
source "{RELEASE_COMMON}"
host=raptor
{_between(BUILD, ROLES_BLOCK)}
spark_role_label="$1"
spark_tp_roles="$2"
{_between(BUILD, POSTCHECK_BLOCK)}
echo ACCEPTED
"""
    for advertised, requested in [
        (UNIVERSAL, UNIVERSAL), ("tp3;tp6;tp2", UNIVERSAL),
        ("tp2;tp6", "tp2;tp6"), ("tp6", "tp6"), ("", ""),
    ]:
        result = _bash(harness, positional=[advertised, requested])
        assert result.returncode == 0, (advertised, requested, result.stderr)
        assert "ACCEPTED" in result.stdout

    for advertised, requested in [
        ("tp2;tp3", UNIVERSAL),            # partial export, full claim
        ("tp2;tp3;tp6;tp4", UNIVERSAL),    # foreign token in the label
        ("tp2;tp3;tp6", "tp2;tp3"),        # label overstates the request
        (UNIVERSAL, ""),                   # a legacy build that baked roles
        ("tp5", UNIVERSAL),                # a label nothing can honor
    ]:
        result = _bash(harness, positional=[advertised, requested])
        assert result.returncode == 2, (advertised, requested, result.stdout)
        assert "expected exactly" in result.stderr, result.stderr
        assert advertised in result.stderr


def test_docker_exported_role_manifest_is_checksummed_per_role():
    """Both roles' built-role manifest must be exported and hashed."""
    text = BUILD.read_text(encoding="utf-8")
    block = text.split("sha256sum \\", 1)[1].split(">SHA256SUMS")[0]
    for role in ("coordinator", "spark-expert"):
        assert f"{role}/V41_EXPERT_TP_AOT.json" in block, role
    assert 'docker cp "$coordinator_container:/opt/ds41rt/share/V41_EXPERT_TP_AOT.json"' in text
    assert 'docker cp "$container:/opt/ds41rt/share/V41_EXPERT_TP_AOT.json"' in text


# ---------------------------------------------------------------------------
# push-containers.sh: a published release pair declares what it can serve
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("advertised,publishable", [
    (UNIVERSAL, True),
    ("tp6;tp2;tp3", True),           # a permutation is the same coverage
    ("tp2;tp3;tp6;tp4", True),       # more than required still covers it
    ("tp6", False),                  # single-topology subset
    ("tp2;tp3", False),              # missing tp6
    ("tp2 tp3 tp6", False),          # wrong separator: advertises nothing
    ("garbage", False),
    ("", False),
])
def test_publisher_requires_the_universal_role_membership(advertised, publishable):
    """A published release pair must serve every approved native topology."""
    result = _bash(f"""
set -euo pipefail
source "{RELEASE_COMMON}"
{_between(PUSH, GUARD_BLOCK)}
push_require_universal_roles '{advertised}' ghcr.io/tpurtell/ds41rt-spark-expert:v10
echo PUBLISHABLE
""")
    assert (result.returncode == 0) == publishable, (advertised, result.stdout, result.stderr)
    assert ("PUBLISHABLE" in result.stdout) == publishable
    if not publishable:
        assert "refusing to publish a legacy or subset build" in result.stderr, result.stderr


def test_publisher_names_the_first_missing_role():
    result = _bash(f"""
set -euo pipefail
source "{RELEASE_COMMON}"
{_between(PUSH, GUARD_BLOCK)}
push_require_universal_roles 'tp2;tp3' ghcr.io/tpurtell/ds41rt-spark-expert:v10
""")
    assert result.returncode == 2
    assert "does not advertise Spark expert role 'tp6'" in result.stderr, result.stderr


def test_publisher_guard_runs_before_the_image_is_retagged():
    text = PUSH.read_text(encoding="utf-8")
    invoked = text.index('push_require_universal_roles "$spark_roles"')
    assert invoked < text.index("docker tag "), "guard must run before any retag"
    assert invoked < text.index("docker push"), "guard must run before any push"


def test_publisher_reads_spark_labels_over_ssh():
    """The Spark image lives on SPARK_0_HOST, not in the local daemon."""
    text = PUSH.read_text(encoding="utf-8")
    read = text.split('spark_roles="$(')[1].split('\n)"')[0]
    assert "ssh -o BatchMode=yes" in read and "docker image inspect" in read
    assert 'io.ds41rt.v41.spark_tp_roles' in read


# ---------------------------------------------------------------------------
# docker/Dockerfile.release: structure plus the real bake-time jq corroboration
# ---------------------------------------------------------------------------

INSTRUCTION = re.compile(
    r"^(ADD|ARG|CMD|COPY|ENTRYPOINT|ENV|EXPOSE|FROM|HEALTHCHECK|LABEL|MAINTAINER|"
    r"ONBUILD|RUN|SHELL|STOPSIGNAL|USER|VOLUME|WORKDIR)\b")


def _logical_lines() -> list[str]:
    """Join physical continuations the way the Dockerfile parser does."""
    logical: list[str] = []
    buffer = ""
    for physical in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        stripped = physical.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        logical.append((buffer + stripped).strip())
        buffer = ""
    assert not buffer, "Dockerfile.release ends with a dangling continuation"
    return logical


def test_dockerfile_lines_parse_as_instructions():
    for line in _logical_lines():
        assert INSTRUCTION.match(line), f"unparsable Dockerfile line: {line[:70]}"
        for quote in ("'", '"'):
            assert line.count(quote) % 2 == 0, f"unbalanced {quote} in: {line[:70]}"


def test_role_build_arg_keeps_its_empty_default():
    """build.sh resolves the roles; the Dockerfile must not bake its own default.

    A non-empty ARG default would label a coordinator image with Spark expert
    roles the coordinator build cannot contain, and would silently override an
    operator's explicit subset request.
    """
    assert "ARG DS41RT_V41_SPARK_TP_ROLES=\n" in DOCKERFILE.read_text(encoding="utf-8")


def _corroboration_line() -> str:
    matches = [line for line in _logical_lines()
               if "V41_EXPERT_TP_AOT.json" in line and "jq -e" in line]
    assert len(matches) == 1, matches
    line = matches[0]
    assert line.startswith("RUN for exl3_family"), (
        "the corroboration belongs in the existing verification layer, not a new one")
    # The expected hash comes from the manifest, the actual hash from the library
    # that ships, and both are compared inside one jq expression.
    assert '--arg lib "$(sha256sum /opt/ds41rt/lib/libds41rt_native.so | cut -d" " -f1)"' in line
    assert "$doc.native_library_sha256 == $lib" in line
    return line


def _jq_filter() -> str:
    segment = _corroboration_line().split("jq -e --arg role", 1)[1]
    filters = re.findall(r"'([^']*)'", segment)
    assert filters, segment
    return filters[0]


def _manifest(role: str, roles: list[str], library_sha: str) -> str:
    return json.dumps({
        "schema": 1,
        "role": role,
        "requested": ";".join(roles),
        "spark_tp_roles": roles,
        "native_library_sha256": library_sha,
        "symbols_verified": bool(roles),
        "symbol_verification_tool": "nm" if roles else None,
        "manifests": {r: {"spark_tp_degree": int(r[2:])} for r in roles},
    })


def _jq(role: str, claimed: str, document: str, library_sha: str) -> bool:
    """Run the Dockerfile's real jq corroboration.

    `library_sha` is the hash of the library that ships in the image, supplied
    independently of the manifest so the comparison cannot be a tautology.
    """
    result = subprocess.run(
        ["jq", "-e", "--arg", "role", role, "--arg", "claimed", claimed,
         "--arg", "lib", library_sha, _jq_filter()],
        input=document, capture_output=True, text=True, timeout=60, check=False)
    assert "jq: error" not in result.stderr, result.stderr
    assert result.returncode in (0, 1, 5), (result.returncode, result.stderr)
    return result.returncode == 0


def test_bake_check_accepts_the_real_coordinator_and_expert_shapes():
    assert _jq("coordinator", "", _manifest("coordinator", [], "coord-sha"), "coord-sha")
    assert _jq("expert", UNIVERSAL,
               _manifest("expert", ["tp2", "tp3", "tp6"], "spark-sha"), "spark-sha")
    assert _jq("expert", "tp6;tp2;tp3",
               _manifest("expert", ["tp2", "tp3", "tp6"], "spark-sha"), "spark-sha")
    assert _jq("expert", "tp6", _manifest("expert", ["tp6"], "spark-sha"), "spark-sha")
    assert _jq("expert", "", _manifest("expert", [], "spark-sha"), "spark-sha")


def test_bake_check_rejects_a_claim_the_artifacts_do_not_corroborate():
    assert not _jq("expert", UNIVERSAL, _manifest("expert", ["tp2", "tp3"], "sha"), "sha")
    assert not _jq("expert", "tp2", _manifest("expert", ["tp2", "tp3"], "sha"), "sha")
    # An empty claim over a manifest that recorded roles is still a mismatch.
    assert not _jq("expert", "", _manifest("expert", ["tp2"], "sha"), "sha")
    # The shipped library is not the one the manifest was written from.
    assert not _jq("expert", UNIVERSAL,
                   _manifest("expert", ["tp2", "tp3", "tp6"], "sha"), "other-sha")
    assert not _jq("coordinator", UNIVERSAL, _manifest("coordinator", [], "sha"), "sha")
    assert not _jq("expert", "", _manifest("coordinator", [], "sha"), "sha")


def test_release_builds_label_only_the_expert_image():
    """The expert image is labelled from the resolved set; the coordinator is not.

    Spark expert roles are expert-only: `scripts/build-release-artifacts.sh`
    rejects them for `role=coordinator`, so the coordinator build passes no role
    value and inherits the empty `ARG` default pinned by
    `test_role_build_arg_keeps_its_empty_default`.
    """
    text = BUILD.read_text(encoding="utf-8")
    # One resolved set for the expert image, one explicit empty for the
    # coordinator, and nothing else that can drift.
    assert text.count('--build-arg DS41RT_V41_SPARK_TP_ROLES="$spark_tp_roles"') == 1
    assert text.count('--build-arg DS41RT_V41_SPARK_TP_ROLES= ' + chr(92)) == 1
    assert text.count("DS41RT_V41_SPARK_TP_ROLES=") == 2
