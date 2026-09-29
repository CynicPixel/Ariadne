#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# AWS CLI and Terraform, run from pinned official images. Nothing is installed
# on the host; Docker is the only dependency.
#
#   infra/scripts/aws.sh login                   # sign in (browser on any device)
#   infra/scripts/aws.sh aws sts get-caller-identity
#   infra/scripts/aws.sh tf bootstrap init       # terraform in infra/aws/terraform/bootstrap
#   infra/scripts/aws.sh tf main plan            # terraform in infra/aws/terraform
#
# CREDENTIALS. `aws login` (CLI >= 2.32) exchanges a console sign-in for
# short-lived credentials that refresh automatically; no access keys exist
# anywhere. `--remote` prints a URL and asks for the code shown after sign-in,
# because a container has no browser and no reachable localhost callback.
#
# Terraform's AWS provider cannot read the `aws login` cache directly
# (terraform-provider-aws #45316), so for `tf` this script asks the CLI
# container to export the CURRENT short-lived credentials and hands them to the
# Terraform container through a mode-600 env file that is deleted on exit. The
# values never appear on a command line or in `ps`.
#
# The host's ~/.aws is mounted, and both containers run as the invoking user,
# so the login cache stays owned by you rather than by root.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TF_ROOT="$REPO_ROOT/infra/aws/terraform"

# Pinned by digest (resolved 2026-09-29). A tag is not a version.
AWS_IMAGE="amazon/aws-cli:2.37.5@sha256:fb7ccfc7b4e3a05017e6c9ded4ba959b99af622130e211fb976c68c2f4d3f224"
TF_IMAGE="hashicorp/terraform:1.16.0@sha256:64360659224d6cbeb099eeed61aa66a80e02c18ba08c0243bd905165b47b088e"

export AWS_REGION="${AWS_REGION:-eu-north-1}"
AWS_PROFILE_NAME="${AWS_PROFILE:-default}"

mkdir -p "$HOME/.aws" "$REPO_ROOT/build/tf-plugin-cache"

tty_flags=(-i)
[ -t 0 ] && [ -t 1 ] && tty_flags+=(-t)

user_flags=(--user "$(id -u):$(id -g)" -e HOME=/home/op)

run_aws() {
  docker run --rm "${tty_flags[@]}" "${user_flags[@]}" \
    -v "$HOME/.aws":/home/op/.aws \
    -v "$REPO_ROOT":/work -w /work \
    -e AWS_REGION -e AWS_PROFILE="$AWS_PROFILE_NAME" \
    "$AWS_IMAGE" "$@"
}

run_tf() {
  local stack="$1"; shift
  local dir
  case "$stack" in
    main)      dir="$TF_ROOT" ;;
    bootstrap) dir="$TF_ROOT/bootstrap" ;;
    *) echo "unknown stack '$stack' (expected: main | bootstrap)" >&2; exit 2 ;;
  esac
  local rel="${dir#"$REPO_ROOT"/}"
  [ "$dir" = "$REPO_ROOT" ] && rel="."

  # EXIT trap at script scope, not RETURN: this script runs one command per
  # invocation, and EXIT is the only trap whose timing is unambiguous.
  envfile="$(mktemp)"
  chmod 600 "$envfile"
  trap 'rm -f "$envfile"' EXIT

  # `validate`, `fmt`, and `init -backend=false` need no credentials;
  # everything else does. Failing to export is fatal rather than silently
  # running Terraform with no identity.
  local needs_creds=1
  case "${1:-}" in
    validate|fmt|version) needs_creds=0 ;;
    init) for a in "$@"; do [ "$a" = "-backend=false" ] && needs_creds=0; done ;;
  esac
  case "$needs_creds" in
    0) : ;;
    *)
      if ! run_aws configure export-credentials --format env-no-export \
             > "$envfile" 2>/dev/null; then
        echo "No valid AWS credentials. Run: infra/scripts/aws.sh login" >&2
        exit 1
      fi
      ;;
  esac
  echo "AWS_REGION=$AWS_REGION" >> "$envfile"

  docker run --rm "${tty_flags[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp \
    --env-file "$envfile" \
    -e TF_PLUGIN_CACHE_DIR=/work/build/tf-plugin-cache \
    -e TF_IN_AUTOMATION=1 \
    -v "$REPO_ROOT":/work -w "/work/$rel" \
    "$TF_IMAGE" "$@"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  login) run_aws login --remote "$@" ;;
  aws)   run_aws "$@" ;;
  tf)    run_tf "$@" ;;
  *)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
