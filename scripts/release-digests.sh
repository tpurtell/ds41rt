#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/release-common.sh"

usage() {
  cat <<'EOF'
Usage: scripts/release-digests.sh capture [--config FILE] [--tag TAG]
                                         [--evidence FILE] [--force]
       scripts/release-digests.sh verify  [--config FILE] [--tag TAG]
                                         --evidence FILE
                                         [--role coordinator|spark|all]

Records and re-checks the registry digests of one published DS41RT image pair.
It is read-only and never tags, pushes or edits a configuration: publication
stays in ./push-containers.sh.

capture  Resolve the coordinator and Spark image references from --config
         (default: ds41rt.config), obtain one anonymous GHCR pull token per
         repository, and record the Docker-Content-Digest of the tag. The
         coordinator is an OCI image index and the Spark expert is a single
         manifest, so each role's own digest is recorded. --evidence writes the
         capture to FILE (refused if it already exists unless --force).
         --tag records another tag of the same repositories (default: the tag
         named by the configuration); use it for the pre-push `latest` rollback
         baseline.

verify   Pull the same references with a throwaway empty DOCKER_CONFIG, so no
         host credential can be consulted, and require the pull's reported
         Digest to equal the captured value. Run it on a host that can reach
         GHCR; --role selects one registration (default: both). The requested
         tag must match the evidence, so a capture cannot be replayed for
         another tag.

The two GHCR repositories are the ones ./push-containers.sh publishes to. The
tag comes from the configuration; after the v10 runtime promotion ds41rt.config
itself names the v10 pair, and the explicit build target is identical:
  scripts/release-digests.sh capture --config ds41rt.config \
      --evidence /path/to/v10-evidence/digests.env
  scripts/release-digests.sh verify  --config ds41rt.config \
      --evidence /path/to/v10-evidence/digests.env
EOF
}

digest_pattern='^sha256:[0-9a-f]{64}$'

# Split a tagged ghcr.io reference into a repository line and a tag line, or die.
# The validation is deliberately narrow: these values reach curl URLs, so a
# repository that is not a ghcr.io path or a tag carrying a slash is refused
# rather than escaped.
release_digests_resolve_reference() {
  local ref="$1" name="$2"
  local repository="${ref%:*}"
  local tag="${ref##*:}"
  [[ "$ref" == ghcr.io/*/* && "$repository" != "$ref" && -n "$tag" && "$tag" != */* ]] ||
    release_die "$name must be a tagged ghcr.io image reference, got: $ref"
  [[ "${repository#ghcr.io/}" != *:* ]] ||
    release_die "$name must not carry a registry port: $ref"
  printf '%s\t%s\n' "$repository" "$tag"
}

release_digests_split() {
  # $1 as printed by release_digests_resolve_reference, into RELEASE_DIGEST_REPOSITORY
  # and RELEASE_DIGEST_TAG.
  IFS=$'\t' read -r RELEASE_DIGEST_REPOSITORY RELEASE_DIGEST_TAG <<<"$1"
}

# One anonymous pull token for REPOSITORY, as GHCR's token endpoint returns it.
ghcr_pull_token() {
  local repository="$1" path
  path="${repository#ghcr.io/}"
  [[ -n "$path" ]] || release_die "not a ghcr.io repository: $repository"
  curl -fsS "https://ghcr.io/token?scope=repository:${path}:pull&service=ghcr.io" |
    jq -er '.token | select(type == "string" and length > 0)' ||
    release_die "could not obtain an anonymous pull token for $repository"
}

# The registry's own digest for REPOSITORY:TAG, from the manifest response header.
# It is the Docker-Content-Digest, not a local image id or config digest.
ghcr_manifest_digest() {
  local repository="$1" tag="$2" token="$3" path headers
  path="${repository#ghcr.io/}"
  headers="$(
    curl -fsSI \
      -H "Authorization: Bearer ${token}" \
      -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json' \
      "https://ghcr.io/v2/${path}/manifests/${tag}"
  )" || release_die "registry manifest request failed for ${repository}:${tag}"
  awk 'BEGIN { IGNORECASE = 1 }
       /^docker-content-digest:/ {
         value = $2
         gsub(/\r/, "", value)
         print value
         exit
       }' <<<"$headers"
}

# Print the repository/tag/digest evidence lines for one role.
release_digests_capture_role() {
  local role="$1" ref="$2" resolved repository tag token digest
  resolved="$(release_digests_resolve_reference "$ref" "$role image")"
  release_digests_split "$resolved"
  repository="$RELEASE_DIGEST_REPOSITORY"
  tag="${tag_override:-$RELEASE_DIGEST_TAG}"
  token="$(ghcr_pull_token "$repository")"
  digest="$(ghcr_manifest_digest "$repository" "$tag" "$token")"
  [[ "$digest" =~ $digest_pattern ]] ||
    release_die "registry did not report a sha256 digest for ${repository}:${tag} (got: '${digest:-<empty>}')"
  printf '%s.repository=%s\n%s.tag=%s\n%s.digest=%s\n' \
    "$role" "$repository" "$role" "$tag" "$role" "$digest"
}

release_digests_write_evidence() {
  local path="$1" content="$2" directory temporary
  directory="$(dirname "$path")"
  [[ -d "$directory" ]] || release_die "evidence directory does not exist: $directory"
  if [[ -e "$path" && "$force" != 1 ]]; then
    release_die "evidence file already exists (refusing to overwrite without --force): $path"
  fi
  # Write beside the target and rename, so a partial capture can never replace a
  # complete earlier record.
  temporary="$(mktemp "${directory}/.release-digests.XXXXXX")"
  printf '%s' "$content" >"$temporary"
  mv -f "$temporary" "$path"
}

release_digests_run_capture() {
  release_need curl
  release_need jq
  release_need sha256sum
  local body role ref
  body=""
  for role in coordinator spark; do
    if [[ "$role" == coordinator ]]; then
      ref="$COORDINATOR_DOCKER_INFERENCE"
    else
      ref="$SPARK_EXPERT_DOCKER_INFERENCE"
    fi
    body+="$(release_digests_capture_role "$role" "$ref")"$'\n'
  done
  body+="config=${RELEASE_CONFIG}"$'\n'
  body+="config.sha256=$(sha256sum "$RELEASE_CONFIG" | awk '{print $1}')"$'\n'
  printf '%s' "$body"
  [[ -z "$evidence" ]] || release_digests_write_evidence "$evidence" "$body"
}

release_digests_evidence_value() {
  local path="$1" key="$2" value
  [[ -f "$path" ]] || release_die "evidence file not found: $path"
  value="$(
    awk -F= -v key="$key" 'index($0, key "=") == 1 { v = substr($0, length(key) + 2) }
                            END { if (v != "") print v }' "$path"
  )"
  [[ -n "$value" ]] || release_die "evidence file has no $key: $path"
  printf '%s\n' "$value"
}

# Pull REF with a throwaway empty DOCKER_CONFIG and print the digest the registry
# reported. A credential file appearing in that directory fails the check: the
# point of the exercise is that no host credential was consulted.
release_digests_anonymous_pull() {
  local ref="$1" config_dir output digest
  config_dir="$(mktemp -d "${TMPDIR:-/tmp}/ds41rt-anon-pull.XXXXXX")"
  chmod 700 "$config_dir"
  if ! output="$(DOCKER_CONFIG="$config_dir" docker pull "$ref" 2>&1)"; then
    rm -rf "$config_dir"
    release_die "anonymous pull failed: $ref"
  fi
  digest="$(
    printf '%s\n' "$output" |
      sed -n 's/^Digest:[[:space:]]*\(sha256:[0-9a-f]\{64\}\)[[:space:]]*$/\1/p' |
      tail -n1
  )"
  if [[ -e "$config_dir/config.json" ]]; then
    rm -rf "$config_dir"
    release_die "anonymous pull wrote a credential file; the pull was not anonymous: $config_dir/config.json"
  fi
  rm -rf "$config_dir"
  [[ "$digest" =~ $digest_pattern ]] ||
    release_die "anonymous pull printed no registry digest for $ref"
  printf '%s\n' "$digest"
}

release_digests_run_verify() {
  release_need docker
  [[ -n "$evidence" ]] || release_die "verify requires --evidence FILE"
  local role ref resolved repository tag expected_repository expected_tag expected_digest digest
  for role in "${verify_roles[@]}"; do
    if [[ "$role" == coordinator ]]; then
      ref="$COORDINATOR_DOCKER_INFERENCE"
    else
      ref="$SPARK_EXPERT_DOCKER_INFERENCE"
    fi
    expected_repository="$(release_digests_evidence_value "$evidence" "$role.repository")"
    expected_tag="$(release_digests_evidence_value "$evidence" "$role.tag")"
    expected_digest="$(release_digests_evidence_value "$evidence" "$role.digest")"
    [[ "$expected_digest" =~ $digest_pattern ]] ||
      release_die "evidence $role.digest is not a sha256 digest: $expected_digest"
    resolved="$(release_digests_resolve_reference "$ref" "$role image")"
    release_digests_split "$resolved"
    repository="$RELEASE_DIGEST_REPOSITORY"
    tag="${tag_override:-$RELEASE_DIGEST_TAG}"
    [[ "$repository" == "$expected_repository" && "$tag" == "$expected_tag" ]] ||
      release_die "evidence is for ${expected_repository}:${expected_tag} but ${repository}:${tag} was requested"
    digest="$(release_digests_anonymous_pull "${repository}:${tag}")"
    [[ "$digest" == "$expected_digest" ]] ||
      release_die "anonymous pull digest $digest does not match the captured $expected_digest for ${repository}:${tag}"
    printf 'anonymous pull verified %s %s %s\n' "$role" "${repository}:${tag}" "$digest"
  done
}

mode="${1:-}"
case "$mode" in
  capture|verify) shift ;;
  -h|--help) usage; exit 0 ;;
  "")
    usage >&2
    exit 2
    ;;
  *)
    release_die "unknown release-digests mode: $mode (expected capture or verify)"
    ;;
esac

config="$repo_root/ds41rt.config"
evidence=
force=0
role=all
tag_override=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 && -n "$2" ]] || release_die "--config requires a configuration file"
      config="$2"
      shift 2
      ;;
    --tag)
      [[ $# -ge 2 && -n "$2" ]] || release_die "--tag requires a Docker tag"
      tag_override="$2"
      shift 2
      ;;
    --evidence)
      [[ $# -ge 2 && -n "$2" ]] || release_die "--evidence requires a file path"
      evidence="$2"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    --role)
      [[ $# -ge 2 && -n "$2" ]] || release_die "--role requires coordinator, spark or all"
      role="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      release_die "unknown release-digests argument: $1"
      ;;
  esac
done

case "$role" in
  coordinator|spark) verify_roles=("$role") ;;
  all) verify_roles=(coordinator spark) ;;
  *) release_die "--role must be coordinator, spark or all, got: $role" ;;
esac

if [[ -n "$tag_override" ]]; then
  [[ "$tag_override" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] ||
    release_die "invalid Docker tag: $tag_override"
fi

release_load_config "$config"

case "$mode" in
  capture) release_digests_run_capture ;;
  verify) release_digests_run_verify ;;
esac
