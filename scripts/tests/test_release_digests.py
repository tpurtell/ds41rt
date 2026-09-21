#!/usr/bin/env python3
"""CPU-only tests for `scripts/release-digests.sh`.

The helper is the post-push half of publication: it records the digest the
registry itself reports for the coordinator OCI index and the Spark expert
manifest, then re-checks it with an anonymous pull. It must stay read-only - no
tag, no push, no configuration edit - and it must never consult a host
credential.

No registry, Docker daemon, SSH or host is touched: `curl` and `docker` are
recording stubs on PATH. The curl stub serves the GHCR token JSON and manifest
response headers from a per-repository map, so the assertions are about the
exact URLs, request shape and digest parsing the helper performs.
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "scripts" / "release-digests.sh"
DEFAULT_CONFIG = REPO / "ds41rt.config"
V10_CONFIG = REPO / "ds41rt.build-v10.config"

COORDINATOR_REPO = "ghcr.io/tpurtell/ds41rt-coordinator"
SPARK_REPO = "ghcr.io/tpurtell/ds41rt-spark-expert"
COORDINATOR_PATH = "tpurtell/ds41rt-coordinator"
SPARK_PATH = "tpurtell/ds41rt-spark-expert"

COORDINATOR_DIGEST = "sha256:" + "a" * 64
SPARK_DIGEST = "sha256:" + "b" * 64
OTHER_DIGEST = "sha256:" + "c" * 64

CURL_STUB = textwrap.dedent(
    r"""#!/usr/bin/env bash
    set -euo pipefail
    log="${DS41RT_TEST_CURL_LOG:?DS41RT_TEST_CURL_LOG must be set}"
    line=""
    for token in "$@"; do
      if [[ -z "$line" ]]; then line="$token"; else line="$line"$'\t'"$token"; fi
    done
    printf '%s\n' "$line" >>"$log"
    [[ "${DS41RT_TEST_CURL_FAIL:-0}" != 1 ]] || exit 22
    url=""
    for token in "$@"; do
      case "$token" in
        https://*) url="$token" ;;
      esac
    done
    [[ -n "$url" ]] || exit 2
    case "$url" in
      https://ghcr.io/token*)
        printf '%s\n' '{"token":"anonymous-pull-token"}'
        ;;
      https://ghcr.io/v2/*/manifests/*)
        path="${url#https://ghcr.io/v2/}"
        path="${path%%/manifests/*}"
        digest="$(awk -F= -v key="$path" '$1 == key { print $2 }' "${DS41RT_TEST_DIGESTS:?}" | tail -n1)"
        if [[ -z "$digest" ]]; then
          printf 'HTTP/2 200\r\ncontent-type: application/vnd.oci.image.index.v1+json\r\n\r\n'
          exit 0
        fi
        printf 'HTTP/2 200\r\ncontent-type: application/vnd.oci.image.index.v1+json\r\ndocker-content-digest: %s\r\n\r\n' "$digest"
        ;;
      *)
        exit 2
        ;;
    esac
    """
)

DOCKER_STUB = textwrap.dedent(
    r"""#!/usr/bin/env bash
    set -euo pipefail
    log="${DS41RT_TEST_PULL_LOG:?DS41RT_TEST_PULL_LOG must be set}"
    line=""
    for token in "$@"; do
      if [[ -z "$line" ]]; then line="$token"; else line="$line"$'\t'"$token"; fi
    done
    printf 'DOCKER_CONFIG=%s\t%s\n' "${DOCKER_CONFIG:-<unset>}" "$line" >>"$log"
    if [[ "${DS41RT_TEST_WRITES_CREDENTIAL:-0}" == 1 ]]; then
      : >"${DOCKER_CONFIG}/config.json"
    fi
    ref=""
    for token in "$@"; do
      case "$token" in
        ghcr.io/*) ref="$token" ;;
      esac
    done
    digest="$(awk -F= -v key="$ref" '$1 == key { print $2 }' "${DS41RT_TEST_PULL_DIGESTS:?}" | tail -n1)"
    printf 'Digest: %s\n' "${digest:-sha256:0000000000000000000000000000000000000000000000000000000000000000}"
    """
)


@pytest.fixture
def harness(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("curl", CURL_STUB), ("docker", DOCKER_STUB)):
        path = bin_dir / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    curl_log = tmp_path / "curl.log"
    pull_log = tmp_path / "pull.log"
    digests = tmp_path / "registry-digests.env"
    digests.write_text(
        f"{COORDINATOR_PATH}={COORDINATOR_DIGEST}\n{SPARK_PATH}={SPARK_DIGEST}\n",
        encoding="utf-8",
    )
    pull_digests = tmp_path / "pull-digests.env"
    pull_digests.write_text(
        f"{COORDINATOR_REPO}:v10={COORDINATOR_DIGEST}\n{SPARK_REPO}:v10={SPARK_DIGEST}\n",
        encoding="utf-8",
    )
    environment = {
        "PATH": f"{bin_dir}:/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
        "DS41RT_TEST_CURL_LOG": str(curl_log),
        "DS41RT_TEST_PULL_LOG": str(pull_log),
        "DS41RT_TEST_DIGESTS": str(digests),
        "DS41RT_TEST_PULL_DIGESTS": str(pull_digests),
    }

    def run(*args, env=None):
        merged = dict(environment)
        merged.update(env or {})
        return subprocess.run(
            [str(HELPER), *args], cwd=REPO, capture_output=True, text=True,
            timeout=120, env=merged, check=False,
        )

    def curl_calls():
        if not curl_log.exists():
            return []
        return [line.split("\t") for line in curl_log.read_text().splitlines()]

    def pull_calls():
        if not pull_log.exists():
            return []
        rows = []
        for line in pull_log.read_text().splitlines():
            config, _, rest = line.partition("\t")
            rows.append((config.removeprefix("DOCKER_CONFIG="), rest.split("\t")))
        return rows

    class Harness:
        pass

    fixture = Harness()
    fixture.run = run
    fixture.curl_calls = curl_calls
    fixture.pull_calls = pull_calls
    fixture.tmp_path = tmp_path
    fixture.digests = digests
    fixture.pull_digests = pull_digests
    return fixture


def _capture(harness, *extra, env=None):
    evidence = harness.tmp_path / "digests.env"
    result = harness.run(
        "capture", "--config", str(V10_CONFIG), "--evidence", str(evidence),
        *extra, env=env,
    )
    return result, evidence


def _captured_evidence(harness):
    result, evidence = _capture(harness)
    assert result.returncode == 0, result.stdout + result.stderr
    return evidence


def test_capture_records_the_registry_digests_and_writes_evidence(harness):
    evidence = _captured_evidence(harness)
    content = evidence.read_text(encoding="utf-8")
    assert f"coordinator.repository={COORDINATOR_REPO}\n" in content
    assert "coordinator.tag=v10\n" in content
    assert f"coordinator.digest={COORDINATOR_DIGEST}\n" in content
    assert f"spark.repository={SPARK_REPO}\n" in content
    assert "spark.tag=v10\n" in content
    assert f"spark.digest={SPARK_DIGEST}\n" in content
    assert f"config={V10_CONFIG}\n" in content
    assert "config.sha256=" in content
    assert "config.sha256=<" not in content
    # The tag comes from the configuration, never from a local image id.
    assert "sha256:" + "a" * 64 in content


def test_capture_is_read_only_and_anonymous(harness):
    result, _ = _capture(harness)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = harness.curl_calls()
    urls = [call[-1] for call in calls]
    assert urls[0] == f"https://ghcr.io/token?scope=repository:{COORDINATOR_PATH}:pull&service=ghcr.io"
    assert urls[1] == f"https://ghcr.io/v2/{COORDINATOR_PATH}/manifests/v10"
    assert urls[2] == f"https://ghcr.io/token?scope=repository:{SPARK_PATH}:pull&service=ghcr.io"
    assert urls[3] == f"https://ghcr.io/v2/{SPARK_PATH}/manifests/v10"
    # Manifest reads are HEAD requests carrying the anonymous token and every
    # accepted media type, so the index digest of the coordinator is reported.
    manifest = calls[1]
    assert any("Authorization: Bearer anonymous-pull-token" == token for token in manifest)
    accept = [token for token in manifest if token.startswith("Accept: ")]
    assert accept and "application/vnd.oci.image.index.v1+json" in accept[0]
    assert "-X" not in manifest and "POST" not in manifest and "PUT" not in manifest
    assert "DELETE" not in manifest
    assert harness.pull_calls() == [], "capture must not pull or touch a daemon"


def test_capture_refuses_to_overwrite_evidence_without_force(harness):
    evidence = harness.tmp_path / "digests.env"
    evidence.write_text("previous capture\n", encoding="utf-8")
    result = harness.run(
        "capture", "--config", str(V10_CONFIG), "--evidence", str(evidence)
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "refusing to overwrite" in result.stderr
    assert evidence.read_text(encoding="utf-8") == "previous capture\n"
    result = harness.run(
        "capture", "--config", str(V10_CONFIG), "--evidence", str(evidence), "--force"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert COORDINATOR_DIGEST in evidence.read_text(encoding="utf-8")


def test_capture_refuses_a_registry_response_without_a_digest(harness):
    # Drop the coordinator entry: the header arrives without Docker-Content-Digest.
    harness.digests.write_text(f"{SPARK_PATH}={SPARK_DIGEST}\n", encoding="utf-8")
    evidence = harness.tmp_path / "digests.env"
    result = harness.run(
        "capture", "--config", str(V10_CONFIG), "--evidence", str(evidence)
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "did not report a sha256 digest" in result.stderr
    assert not evidence.exists(), "a partial capture must not be recorded"


def test_capture_fails_closed_when_the_registry_is_unreachable(harness):
    evidence = harness.tmp_path / "digests.env"
    result = harness.run(
        "capture", "--config", str(V10_CONFIG), "--evidence", str(evidence),
        env={"DS41RT_TEST_CURL_FAIL": "1"},
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert not evidence.exists()


def test_verify_uses_a_fresh_anonymous_config_and_matches(harness):
    evidence = _captured_evidence(harness)
    result = harness.run(
        "verify", "--config", str(V10_CONFIG), "--evidence", str(evidence)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("anonymous pull verified") == 2
    calls = harness.pull_calls()
    assert [call[1] for call in calls] == [
        ["pull", f"{COORDINATOR_REPO}:v10"],
        ["pull", f"{SPARK_REPO}:v10"],
    ], calls
    for config_dir, _ in calls:
        assert config_dir != str(harness.tmp_path / ".docker"), config_dir
        assert Path(config_dir).name.startswith("ds41rt-anon-pull."), config_dir
        assert not (Path(config_dir) / "config.json").exists(), config_dir


def test_verify_fails_when_the_pull_digest_differs(harness):
    evidence = _captured_evidence(harness)
    harness.pull_digests.write_text(
        f"{COORDINATOR_REPO}:v10={OTHER_DIGEST}\n{SPARK_REPO}:v10={SPARK_DIGEST}\n",
        encoding="utf-8",
    )
    result = harness.run(
        "verify", "--config", str(V10_CONFIG), "--evidence", str(evidence)
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "does not match the captured" in result.stderr


def test_verify_rejects_a_pull_that_wrote_a_credential_file(harness):
    evidence = _captured_evidence(harness)
    result = harness.run(
        "verify", "--config", str(V10_CONFIG), "--evidence", str(evidence),
        env={"DS41RT_TEST_WRITES_CREDENTIAL": "1"},
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "was not anonymous" in result.stderr


def test_verify_refuses_evidence_for_another_tag(harness):
    evidence = _captured_evidence(harness)
    result = harness.run(
        "verify", "--config", str(DEFAULT_CONFIG), "--evidence", str(evidence)
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "was requested" in result.stderr


def test_capture_records_the_pre_push_latest_baseline_for_rollback(harness):
    """latest moves on push; its pre-push digest is the rollback boundary."""
    harness.digests.write_text(
        f"{COORDINATOR_PATH}={OTHER_DIGEST}\n{SPARK_PATH}={SPARK_DIGEST}\n",
        encoding="utf-8",
    )
    evidence = harness.tmp_path / "pre-push-latest.env"
    result = harness.run(
        "capture", "--config", str(DEFAULT_CONFIG), "--tag", "latest",
        "--evidence", str(evidence),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    content = evidence.read_text(encoding="utf-8")
    assert "coordinator.tag=latest\n" in content
    assert f"coordinator.digest={OTHER_DIGEST}\n" in content
    urls = [call[-1] for call in harness.curl_calls()]
    assert f"https://ghcr.io/v2/{COORDINATOR_PATH}/manifests/latest" in urls


def test_verify_accepts_an_explicit_matching_tag(harness):
    harness.pull_digests.write_text(
        f"{COORDINATOR_REPO}:latest={COORDINATOR_DIGEST}\n"
        f"{SPARK_REPO}:latest={SPARK_DIGEST}\n",
        encoding="utf-8",
    )
    evidence = harness.tmp_path / "pre-push-latest.env"
    capture = harness.run(
        "capture", "--config", str(V10_CONFIG), "--tag", "latest",
        "--evidence", str(evidence),
    )
    assert capture.returncode == 0, capture.stdout + capture.stderr
    result = harness.run(
        "verify", "--config", str(V10_CONFIG), "--tag", "latest",
        "--evidence", str(evidence),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert [call[1] for call in harness.pull_calls()] == [
        ["pull", f"{COORDINATOR_REPO}:latest"],
        ["pull", f"{SPARK_REPO}:latest"],
    ]


@pytest.mark.parametrize("tag", ["bad/tag", "", "-leading"])
def test_an_explicit_tag_is_validated(harness, tag):
    result = harness.run(
        "capture", "--config", str(V10_CONFIG), "--tag", tag,
        "--evidence", str(harness.tmp_path / "digests.env"),
    )
    assert result.returncode == 2, (tag, result.stdout, result.stderr)
    assert harness.curl_calls() == []


def test_verify_can_select_one_role(harness):
    evidence = _captured_evidence(harness)
    result = harness.run(
        "verify", "--config", str(V10_CONFIG), "--evidence", str(evidence),
        "--role", "coordinator",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert [call[1] for call in harness.pull_calls()] == [
        ["pull", f"{COORDINATOR_REPO}:v10"]
    ]


@pytest.mark.parametrize("args,fragment", [
    (("verify",), "requires --evidence"),
    (("verify", "--config", str(V10_CONFIG), "--evidence", "/nonexistent/digests.env"),
     "evidence file not found"),
    (("verify", "--config", str(V10_CONFIG), "--evidence", "x", "--role", "bogus"),
     "--role must be coordinator, spark or all"),
    (("capture", "--config", str(V10_CONFIG), "--role", "bogus"),
     "--role must be coordinator, spark or all"),
    (("capture", "--config"), "--config requires a configuration file"),
])
def test_usage_errors_exit_two(harness, args, fragment):
    result = harness.run(*args)
    assert result.returncode == 2, (args, result.stdout, result.stderr)
    assert fragment in result.stderr, (args, result.stderr)


def test_help_and_unknown_mode(harness):
    result = harness.run("--help")
    assert result.returncode == 0, result.stderr
    assert "capture" in result.stdout and "verify" in result.stdout
    result = harness.run("--help", "--nonsense")
    assert result.returncode == 0, result.stderr
    result = harness.run("publish")
    assert result.returncode == 2
    assert "unknown release-digests mode" in result.stderr
    result = harness.run()
    assert result.returncode == 2
    assert harness.curl_calls() == []
    assert harness.pull_calls() == []


def test_the_helper_never_tags_pushes_or_rewrites_a_config():
    text = HELPER.read_text(encoding="utf-8")
    for forbidden in ("docker push", "docker tag", 'docker image inspect "$1"'):
        assert forbidden not in text, forbidden
    # HTTP use is read-only: only GET/HEAD, no registry write verbs.
    assert "curl -X" not in text and " -X POST" not in text and " -X PUT" not in text
    assert "DELETE" not in text
