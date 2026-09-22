#!/usr/bin/env bash
# Read-only release-artifact verification for one DS41RT release.
#
# This is the reusable form of the throwaway `runs/vN-release/build/10-verify.sh`
# used for v10. It records expected image labels, per-host fleet identity, the
# build's dist inventory and integrity, the V41 expert/TP AOT manifest, EXL3
# package verification and the TP3 rank geometry, then exits non-zero when any
# hard check fails. It never builds, tags, pushes, pulls or edits a config.
#
# Usage:
#   scripts/release/verify-release-artifacts.sh --config ds41rt.build-v11.config \
#     --evidence runs/v11-release/build [--dist PATH] [--require-hosts CSV]
#
# Required checks are never skipped: the v10 pipeline asserted the pair identity
# and identical Spark image id on every one of its four build hosts
# (`runs/v10-release/build/10-verify.sh`, HOSTS=(ostrich dodo emu kiwi)), so by
# default `--require-hosts` is the config's first four SPARK_*_HOST values and
# each must carry the image, reachable, with the identical image id. The
# optional fifth/sixth hosts are scanned too (default: all configured hosts) but
# only report presence, because the release is a four-Spark deployment.
#
# `--fleet-file` replays a previously recorded `10-fleet-spark-images.txt`
# instead of contacting hosts, so the required-host logic is itself testable
# offline. It is hidden from `--help` on purpose: a release verification must
# scan live hosts.
#
# In-image versus dist comparison: the images bind their artifacts at build time
# and are not guaranteed to expose the dist tree; `--dist` enables the package,
# provenance, SHA256SUMS and AOT-manifest checks against the build output. The
# native-library SHA-256 is always compared against the value recorded in the
# dist AOT manifest when both are available.
#
# Exit status: 0 when every hard check passes; 1 when any hard check fails or a
# required input is missing; 2 on usage errors. SKIP is reserved for checks that
# genuinely cannot apply (a Spark role label is only inspectable when a Spark
# image is present, and the dist checks need --dist); the required four-host
# identity check must produce PASS or FAIL, never SKIP.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
probe="$repo_root/scripts/release/image-identity-probe.sh"
required_count=4
max_scan=6

usage() {
  cat <<'EOF'
Usage: scripts/release/verify-release-artifacts.sh --config FILE [options]

  --config FILE         release build config whose pair is verified (required)
  --evidence DIR        evidence directory (required; created if missing)
  --dist PATH           build dist/ tree for package/provenance/SHA256SUMS checks
  --require-hosts CSV   hosts that must carry the Spark image (default: the
                        config's first four SPARK_*_HOST values; the v10 gate
                        required all four). Also accepted as --hosts.
  --help                this message

Read-only. Writes NN-*.txt artifacts under --evidence and prints a summary.
EOF
}

die() { echo "verify-release-artifacts: $*" >&2; exit 2; }

config=""
evidence=""
dist=""
required_csv=""
scan_csv=""
fleet_file=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)        [[ $# -ge 2 ]] || die "--config requires a file";  config="$2"; shift 2 ;;
    --evidence)      [[ $# -ge 2 ]] || die "--evidence requires a dir";  evidence="$2"; shift 2 ;;
    --dist)          [[ $# -ge 2 ]] || die "--dist requires a path";     dist="$2"; shift 2 ;;
    --require-hosts|--hosts) [[ $# -ge 2 ]] || die "$1 requires a list"; required_csv="$2"; shift 2 ;;
    --scan-hosts)    [[ $# -ge 2 ]] || die "--scan-hosts requires a list"; scan_csv="$2"; shift 2 ;;
    --fleet-file)    [[ $# -ge 2 ]] || die "--fleet-file requires a path"; fleet_file="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$config" ]] || { usage >&2; exit 2; }
[[ -n "$evidence" ]] || { usage >&2; exit 2; }

source "$repo_root/scripts/release-common.sh"
release_load_config "$config"

coordinator="$COORDINATOR_DOCKER_INFERENCE"
spark="$SPARK_EXPERT_DOCKER_INFERENCE"
release_version="${coordinator##*:}"
mkdir -p "$evidence" || die "cannot create evidence directory: $evidence"

configured_hosts=()
for i in $(seq 0 $((max_scan - 1))); do
  name="SPARK_${i}_HOST"
  [[ -n "${!name:-}" ]] && configured_hosts+=("${!name}")
done

if [[ -z "$required_csv" ]]; then
  required_csv="$(IFS=,; echo "${configured_hosts[*]:0:$required_count}")"
fi
if [[ -z "$scan_csv" ]]; then
  scan_csv="$(IFS=,; echo "${configured_hosts[*]}")"
fi
IFS=',' read -r -a required_hosts <<<"$required_csv"
IFS=',' read -r -a hosts <<<"$scan_csv"
((${#required_hosts[@]} > 0)) || die "no required hosts resolved"
((${#hosts[@]} > 0)) || die "no hosts to scan"

failures=0
declare -a results=()
record() { # status name detail
  results+=("$1 $2 ${3:-}")
  [[ "$1" == "FAIL" ]] && failures=$((failures + 1))
  return 0
}

echo "== verify-release-artifacts: tag=$release_version evidence=$evidence =="
echo "   coordinator=$coordinator"
echo "   spark=$spark"
echo "   required hosts (must carry the image): $required_csv"
echo "   scanned hosts: $scan_csv"

########################################################################
# 1. coordinator image (local)
########################################################################
"$probe" "$coordinator" >"$evidence/10-local-coordinator-image.txt" 2>&1 || true
coord_field() { sed -n "s/^$1=//p" "$evidence/10-local-coordinator-image.txt" | head -1; }
coord_status="$(coord_field status)"
exec_rev=""

if [[ "$coord_status" != "present" ]]; then
  record FAIL coordinator.present "local image missing: $coordinator"
else
  exec_rev="$(coord_field label.org.opencontainers.image.revision)"
  [[ -n "$exec_rev" ]] || exec_rev="$(coord_field id)"
  record PASS coordinator.present "$(coord_field id)"
  record "$([[ "$(coord_field architecture)" == amd64 ]] && echo PASS || echo FAIL)" \
    coordinator.arch "expected amd64 got $(coord_field architecture)"
  record "$([[ "$(coord_field label.org.opencontainers.image.version)" == "$release_version" ]] && echo PASS || echo FAIL)" \
    coordinator.version "expected $release_version got $(coord_field label.org.opencontainers.image.version)"
  record "$([[ "$(coord_field label.io.ds41rt.role)" == coordinator ]] && echo PASS || echo FAIL)" \
    coordinator.role "expected coordinator got $(coord_field label.io.ds41rt.role)"
  record "$([[ "$(coord_field label.io.ds41rt.cuda_arch)" == 120 ]] && echo PASS || echo FAIL)" \
    coordinator.cuda_arch "expected 120 got $(coord_field label.io.ds41rt.cuda_arch)"
  record "$([[ -z "$(coord_field label.io.ds41rt.v41.spark_tp_roles)" ]] && echo PASS || echo FAIL)" \
    coordinator.spark_tp_roles "expected empty got $(coord_field label.io.ds41rt.v41.spark_tp_roles)"
  record "$([[ "$exec_rev" != *-dirty-* ]] && echo PASS || echo FAIL)" \
    coordinator.clean_revision "$exec_rev"
  record "$([[ -z "$(coord_field label.io.ds41rt.source-manifest.sha256)" ]] && echo PASS || echo FAIL)" \
    coordinator.source_manifest_absent "$(coord_field label.io.ds41rt.source-manifest.sha256)"
fi

########################################################################
# 2. Spark fleet images (remote read-only scan; required four hosts)
########################################################################
spark_exec_rev=""
spark_present_hosts=()
join_by() { local IFS="$1"; shift; echo "$*"; }

if [[ -n "$fleet_file" ]]; then
  [[ -r "$fleet_file" ]] || die "fleet file not readable: $fleet_file"
  cp "$fleet_file" "$evidence/10-fleet-spark-images.txt" \
    || die "cannot copy fleet file into evidence: $fleet_file"
else
  : >"$evidence/10-fleet-spark-images.txt"
  for host in "${hosts[@]}"; do
    {
      echo "===== $host ====="
      timeout 45 ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
        bash -s -- "$spark" <"$probe" 2>&1 || echo "status=unreachable"
    } >>"$evidence/10-fleet-spark-images.txt" 2>&1
  done
fi

fleet_host_status() {
  awk -F= -v h="===== $1 =====" '$0==h{f=1;next} /^=====/{f=0} f&&$1=="status"{print $2; exit}' \
    "$evidence/10-fleet-spark-images.txt"
}
for host in "${hosts[@]}"; do
  [[ "$(fleet_host_status "$host")" == present ]] && spark_present_hosts+=("$host")
done
fleet_field() {
  awk -F= -v h="===== $1 =====" -v k="$2" '$0==h{f=1;next} /^=====/{f=0} f&&$1==k{print $2; exit}' \
    "$evidence/10-fleet-spark-images.txt"
}
anchor_host=""
for host in "${hosts[@]}"; do
  if [[ "$(fleet_host_status "$host")" == present ]]; then anchor_host="$host"; break; fi
done
for host in "${required_hosts[@]}"; do
  status="$(fleet_host_status "$host")"
  if [[ "$status" != present ]]; then
    detail="no image with tag $release_version"
    [[ "$status" == unreachable ]] && detail="host unreachable over SSH"
    [[ -z "$status" ]] && detail="host not present in the fleet record"
    record FAIL "spark.required_host.$host" "$detail"
  fi
done
if ((${#spark_present_hosts[@]} > 0)); then
  record PASS spark.present "present on ${#spark_present_hosts[@]}/${#hosts[@]} host(s): ${spark_present_hosts[*]}"
fi

if [[ -n "$anchor_host" ]]; then
  spark_exec_rev="$(fleet_field "$anchor_host" label.org.opencontainers.image.revision)"
  spark_arch="$(fleet_field "$anchor_host" architecture)"
  spark_version_label="$(fleet_field "$anchor_host" label.org.opencontainers.image.version)"
  spark_role_label="$(fleet_field "$anchor_host" label.io.ds41rt.role)"
  spark_cuda_label="$(fleet_field "$anchor_host" label.io.ds41rt.cuda_arch)"
  spark_roles="$(fleet_field "$anchor_host" label.io.ds41rt.v41.spark_tp_roles)"
  spark_manifest_label="$(fleet_field "$anchor_host" label.io.ds41rt.source-manifest.sha256)"
  # The coordinator block always records a status line, so its revision is set
  # even when the local image is missing.
  record "$([[ "$spark_exec_rev" == "$exec_rev" ]] && echo PASS || echo FAIL)" \
    pair.same_revision "coordinator=${exec_rev:-<none>} spark=$spark_exec_rev"
  record "$([[ "$spark_arch" == arm64 ]] && echo PASS || echo FAIL)" \
    spark.arch "expected arm64 got ${spark_arch:-<none>}"
  record "$([[ "$spark_version_label" == "$release_version" ]] && echo PASS || echo FAIL)" \
    spark.version "expected $release_version got ${spark_version_label:-<none>}"
  record "$([[ "$spark_role_label" == expert ]] && echo PASS || echo FAIL)" \
    spark.role "expected expert got ${spark_role_label:-<none>}"
  record "$([[ "$spark_cuda_label" == 121 ]] && echo PASS || echo FAIL)" \
    spark.cuda_arch "expected 121 got ${spark_cuda_label:-<none>}"
  record "$([[ "$spark_roles" == "tp2;tp3;tp6" ]] && echo PASS || echo FAIL)" \
    spark.spark_tp_roles "expected tp2;tp3;tp6 got ${spark_roles:-<none>}"
  record "$([[ "$spark_exec_rev" != *-dirty-* ]] && echo PASS || echo FAIL)" \
    spark.clean_revision "$spark_exec_rev"
  record "$([[ -z "$spark_manifest_label" ]] && echo PASS || echo FAIL)" \
    spark.source_manifest_absent "${spark_manifest_label:-<empty>}"
else
  record FAIL spark.identity "no scanned host carries $spark; identity/version/role checks cannot be evaluated"
fi

# Every required host must carry the identical Spark image id, and every scanned
# host that carries the image must agree. This is the v10 four-host gate and it
# must never be SKIP: a missing required host already fails above, and here a
# divergence between present hosts fails too.
required_present=0
required_ids=""
for host in "${required_hosts[@]}"; do
  [[ "$(fleet_host_status "$host")" == present ]] || continue
  required_present=$((required_present + 1))
  required_ids="$required_ids $(fleet_field "$host" id)"
done
unique_required_ids="$(printf '%s\n' $required_ids | sort -u | tr '\n' ' ' | sed 's/ *$//')"
if ((${#required_hosts[@]} > 0)); then
  if ((required_present == ${#required_hosts[@]})) &&
     [[ "$(printf '%s\n' $required_ids | sort -u | wc -l)" == 1 && -n "$unique_required_ids" ]]; then
    record PASS spark.required_hosts_identical_ids \
      "all ${#required_hosts[@]} required host(s) carry $unique_required_ids"
  else
    record FAIL spark.required_hosts_identical_ids \
      "required=${#required_hosts[@]} present=$required_present ids:'$unique_required_ids'"
  fi
fi
all_ids="$(awk -F= '$1=="id" && $2!=""{print $2}' "$evidence/10-fleet-spark-images.txt" | sort -u)"
all_id_count="$(printf '%s\n' "$all_ids" | sed '/^$/d' | wc -l)"
if ((${#spark_present_hosts[@]} >= 2)); then
  record "$([[ "$all_id_count" == 1 ]] && echo PASS || echo FAIL)" \
    spark.fleet_identical_ids "$(printf '%s\n' $all_ids | tr '\n' ' ')"
else
  record SKIP spark.fleet_identical_ids \
    "fewer than two scanned hosts carry the image; the required-host check above is authoritative"
fi

########################################################################
# 3. dist inventory and integrity
########################################################################
if [[ -n "$dist" ]]; then
  {
    echo "# dist inventory"
    echo "dist=$dist"
    if [[ -d "$dist" ]]; then
      find "$dist" -mindepth 1 -maxdepth 3 -type d | sort
      echo "--- sizes ---"
      du -sh "$dist" "$dist/coordinator" "$dist/spark-expert" 2>/dev/null
    else
      echo "status=missing"
    fi
  } >"$evidence/10-dist-inventory.txt" 2>&1
  record "$([[ -d "$dist" ]] && echo PASS || echo FAIL)" dist.present "$dist"

  if [[ -d "$dist" ]]; then
    sparkinfer_rev="$(coord_field label.io.ds41rt.sparkinfer.revision)"
    {
      echo "# dist artifact verification"
      for role in coordinator spark-expert; do
        echo "===== $role ====="
        for family in "$dist/$role"/exl3/exl3-*/; do
          [[ -d "$family" ]] || continue
          echo "-- $(basename "$family") package verify --"
          python3 "$repo_root/python/tools/package_v41_exl3_aot.py" verify \
            --package "$family" --sparkinfer-revision "$sparkinfer_rev"
          echo "package_verify_rc=$?"
        done
        echo "-- sparkinfer provenance --"
        python3 "$repo_root/scripts/sparkinfer-release-provenance.py" \
          --source "$repo_root/third_party/sparkinfer" \
          --lock "$repo_root/third_party/sparkinfer.lock.json" \
          --license "$dist/$role/SPARKINFER_LICENSE" \
          --notices "$dist/$role/THIRD_PARTY_NOTICES.md" \
          --verify "$dist/$role/SPARKINFER_PROVENANCE.json"
        echo "provenance_rc=$?"
        echo "-- shipped checksum lists --"
        (cd "$dist/$role" && sha256sum -c SPARKINFER_SHA256SUMS && sha256sum -c XGRAMMAR_SHA256SUMS)
        echo "shipped_checksums_rc=$?"
        echo "-- V41_EXPERT_TP_AOT.json native_library_sha256 --"
        echo "manifest=$(python3 -c "import json,sys;print(json.load(open('$dist/$role/V41_EXPERT_TP_AOT.json')).get('native_library_sha256'))" 2>/dev/null)"
        echo "actual=$(sha256sum "$dist/$role/libds41rt_native.so" 2>/dev/null | awk '{print $1}')"
      done
    } >"$evidence/10-dist-artifact-verification.txt" 2>&1
    bad_rc="$(grep -c '_rc=[1-9]' "$evidence/10-dist-artifact-verification.txt")"
    record "$([[ "$bad_rc" == 0 ]] && echo PASS || echo FAIL)" \
      dist.artifact_verification "$bad_rc failing step(s); see 10-dist-artifact-verification.txt"
    for role in coordinator spark-expert; do
      manifest="$(python3 -c "import json;print(json.load(open('$dist/$role/V41_EXPERT_TP_AOT.json')).get('native_library_sha256',''))" 2>/dev/null)"
      actual="$(sha256sum "$dist/$role/libds41rt_native.so" 2>/dev/null | awk '{print $1}')"
      record "$([[ -n "$manifest" && "$manifest" == "$actual" ]] && echo PASS || echo FAIL)" \
        "$role.native_library_sha256" "manifest=${manifest:-<none>} actual=${actual:-<none>}"
      roles_json="$(python3 -c "import json;print(';'.join(json.load(open('$dist/$role/V41_EXPERT_TP_AOT.json')).get('spark_tp_roles',[])))" 2>/dev/null)"
      expected_roles=""; [[ "$role" == spark-expert ]] && expected_roles="tp2;tp3;tp6"
      record "$([[ "$roles_json" == "$expected_roles" ]] && echo PASS || echo FAIL)" \
        "$role.aot_spark_tp_roles" "expected '${expected_roles:-<empty>}' got '${roles_json:-<none>}'"
    done

    {
      echo "# dist/SHA256SUMS"
      (cd "$dist" && sha256sum -c SHA256SUMS)
      echo "sha256sums_rc=$?"
    } >"$evidence/10-dist-sha256sums.txt" 2>&1
    record "$([[ "$(tail -1 "$evidence/10-dist-sha256sums.txt")" == "sha256sums_rc=0" ]] && echo PASS || echo FAIL)" \
      dist.sha256sums "see 10-dist-sha256sums.txt"
  fi
else
  record SKIP dist.present "no --dist supplied; package/provenance/SHA256SUMS checks skipped"
fi

########################################################################
# 4. EXL3 TP3 rank geometry, when the purpose-built checker is available
########################################################################
if [[ -n "$dist" && -x "$repo_root/scripts/bench/verify_exl3_tp3.py" ]]; then
  {
    echo "# EXL3 TP3 k23/k34 rank geometry"
    python3 "$repo_root/scripts/bench/verify_exl3_tp3.py" --dist "$dist" \
      --role spark-expert --sparkinfer-revision "$(coord_field label.io.ds41rt.sparkinfer.revision)"
    echo "verify_exl3_tp3_rc=$?"
  } >"$evidence/10-exl3-tp3-verification.txt" 2>&1
  record "$([[ "$(tail -1 "$evidence/10-exl3-tp3-verification.txt")" == "verify_exl3_tp3_rc=0" ]] && echo PASS || echo FAIL)" \
    dist.exl3_tp3_geometry "see 10-exl3-tp3-verification.txt"
else
  record SKIP dist.exl3_tp3_geometry "no --dist or scripts/bench/verify_exl3_tp3.py absent"
fi

########################################################################
# 5. Summary
########################################################################
summary="$evidence/10-verify-summary.txt"
{
  echo "# release artifact verification summary"
  echo "captured_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "config=$config"
  echo "release_tag=$release_version"
  echo "coordinator=$coordinator"
  echo "spark=$spark"
  echo "required_hosts=$required_csv"
  echo "scanned_hosts=$scan_csv"
  echo "dist=${dist:-<none>}"
  echo "--- checks ---"
  printf '%s\n' "${results[@]}"
  echo "--- result ---"
  if ((failures == 0)); then echo "ALL CHECKS PASS"; else echo "FAILURES: $failures"; fi
} >"$summary" 2>&1

cat "$summary"
if ((failures > 0)); then
  echo "verify-release-artifacts: $failures check(s) failed; see $summary" >&2
  exit 1
fi
echo "verify-release-artifacts: all checks passed"
