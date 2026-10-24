#!/bin/bash
# Host side of the generic "build + selected tests" job on a Kaggle CPU kernel (infra/eval/ci_kaggle.sh): the
# off-AWS twin of infra/eval/ci_aws_launch.sh, same NAME / INPUTS / TESTS / EXTRA interface. Moves DEVELOPMENT
# compiles, ctest runs and hardware-agnostic measurements off both the workstation (hard cap: 2 heavy local jobs) and
# the AWS spot quota (32 vCPU) onto free Kaggle batch kernels: 4 vCPU / 32 GiB each, at most 5 at once per account
# (measured 2026-10-24; the 6th push is refused). Results are labelled platform: kaggle-cpu. Phase-gate evidence,
# COOL Arm-vs-Arm and judge-box timing stay on AWS.
#
#   infra/eval/ci_kaggle_launch.sh opencv               once: /opt/opencv5 of ariadne/opencv-base:5.0.0 -> private dataset
#   infra/eval/ci_kaggle_launch.sh stage NAME [COMMIT]  repo at COMMIT (default HEAD) + INPUTS -> private datasets
#   infra/eval/ci_kaggle_launch.sh run NAME             push the kernel, wait, pull -> build/runs-offaws/<id>/
#   infra/eval/ci_kaggle_launch.sh submit NAME          push only (rc 75 when all 5 CPU sessions are busy: retry later)
#   infra/eval/ci_kaggle_launch.sh wait NAME            poll the pushed kernel, then pull
#   infra/eval/ci_kaggle_launch.sh state NAME           one status line of NAME's kernel (RUNNING / COMPLETE / ERROR)
#   infra/eval/ci_kaggle_launch.sh teardown NAME        delete NAME's kernel (datasets are content-addressed and kept)
# NAME: [a-z0-9][a-z0-9-]*, at most 36 chars (it ends up in a Kaggle slug).
#
# Environment (as ci_aws_launch.sh):
#   stage  INPUTS      ignored inputs the job reads, as repo-relative dirs. Default: the five model reference exports
#                      and the PGO oracle (~1 GB, so ctest runs with ARIADNE_REQUIRE_MODELS=1, as the AWS pre-merge).
#                      INPUTS="" stages none. Uploaded once per content hash (ariadne-in-<hash>), reused across runs.
#   run    TESTS       a ctest -R regex; empty = every test; "none" = no ctest (build + EXTRA only)
#          EXTRA       optional shell command, run after the build as uid 1000 from /work with the test environment;
#                      whatever it writes to $CI_OUT comes back as extra_out/
#          MAX_MINUTES the kernel's own timeout (push -t), default 90 (a full ctest takes ~33 min; 3 of 11 kernels on 2026-10-24 stalled RUNNING with no logs, so the Kaggle-side brake must be short. [UNVERIFIED: that push -t is enforced])
#          JOBS / CTEST_JOBS  default 4 / 4 (the kernel has 4 vCPU)
# The repo goes up as `git archive COMMIT` (tracked files only) minus IDEA.md and docs/ARIADNE_PHASED_PLAN.md
# (maintainer, 2026-10-24); everything on Kaggle is PRIVATE. The token never leaves this host: the Kaggle CLI runs
# in kaggle.sh's pinned container with ~/.kaggle mounted read-only.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO_ROOT=$PWD
IMAGE=ariadne-kaggle:2.2.4                  # kaggle.sh's pinned CLI image (built by kaggle.sh on first use)
OPENCV_IMAGE=ariadne/opencv-base:5.0.0
EXCLUDE=(':!IDEA.md' ':!docs/ARIADNE_PHASED_PLAN.md')
DEFAULT_INPUTS=".cache/models/xfeat_ref .cache/models/eigenplaces_ref .cache/models/aliked_lg_ref
  .cache/models/eigenplaces_sim_ref .cache/models/aliked_lg_sim_ref .cache/oracle/pgo"
B=build/ci_kaggle
log() { echo "[ci-kaggle $(TZ=Asia/Kolkata date +%H:%M:%S) IST] $*" >&2; }
die() { log "$*"; exit 1; }

[ -f "$HOME/.kaggle/access_token" ] || [ -f "$HOME/.kaggle/kaggle.json" ] || die "no ~/.kaggle token"
docker image inspect "$IMAGE" >/dev/null 2>&1 || infra/scripts/kaggle.sh whoami >/dev/null
mkdir -p "$B"
kg() {   # the Kaggle CLI as the invoking user; build/ mounted at /b
  docker run --rm -u "$(id -u):$(id -g)" -e HOME=/home/u -e KAGGLE_CONFIG_DIR=/home/u/.kaggle \
    -v "$HOME/.kaggle:/home/u/.kaggle:ro" -v "$REPO_ROOT/build:/b" -w /b "$IMAGE" kaggle "$@"
}
user() { local u; u="$(kg config view 2>/dev/null | sed -n 's/^- username: //p')"
         [ -n "$u" ] && [ "$u" != None ] || die "token resolves to no Kaggle user"; echo "$u"; }

dataset_up() {   # $1 = slug, $2 = dir under build/ holding the files, $3 = title; no-op when it already exists
  local u ds st i; u="$(user)"; ds="$u/$1"
  if kg datasets status "$ds" 2>/dev/null | grep -q ready; then log "dataset $ds exists (content-addressed: reused)"; return; fi
  printf '{"title": "%s", "id": "%s", "licenses": [{"name": "other"}]}\n' "$3" "$ds" > "$REPO_ROOT/build/$2/dataset-metadata.json"
  log "uploading $ds ($(du -sh "$REPO_ROOT/build/$2" | cut -f1))"
  # Private unless --public. Upload files as they are (Kaggle unpacks a .tar it recognises, losing .so symlinks and
  # exec bits: hence the .tar.keep names). Its rc is not trusted: CLI 2.2.4 has printed a JSON parse error after a
  # successful upload (2026-10-24), so the status poll below decides.
  kg datasets create -p "/b/$2" -q > "$REPO_ROOT/build/$2.create.log" 2>&1 || log "create said: $(tail -1 "$REPO_ROOT/build/$2.create.log")"
  for ((i = 0; i < 80; i++)); do
    st="$(kg datasets status "$ds" 2>&1 | tail -1 || true)"   # non-zero while processing: never let set -e end the poll
    case "$st" in *ready*) log "dataset $ds ready"; return ;; esac; sleep 15
  done
  die "dataset $ds not ready after 20 min ($st)"
}

opencv() {
  local id; id="$(docker image inspect "$OPENCV_IMAGE" --format '{{.Id}}')"; id=${id#sha256:}
  local slug="ariadne-opencv5-${id:0:12}" d="$B/opencv-${id:0:12}"
  rm -rf "$d"; mkdir -p "$d"
  docker run --rm -u "$(id -u):$(id -g)" -v "$REPO_ROOT/$d:/o" "$OPENCV_IMAGE" sh -c \
    "printf 'image %s\nimage_id sha256:%s\nopencv_commit %s\n' '$OPENCV_IMAGE' '$id' \
       \"\$(cat /opt/opencv5/opt-opencv-core.sha 2>/dev/null || echo unrecorded)\" > /tmp/kaggle_provenance.txt \
     && tar cf /o/opencv5.tar.keep -C /opt opencv5 -C /tmp kaggle_provenance.txt --transform 's|^kaggle_provenance.txt|opencv5/kaggle_provenance.txt|'"
  echo "$slug" > "$B/opencv_slug"
  dataset_up "$slug" "ci_kaggle/opencv-${id:0:12}" "$slug"
}

NAME="${2:-}"
case "${1:-}" in
  opencv) opencv; exit ;;
  stage|run|submit|wait|state|teardown) ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
esac
[[ "$NAME" =~ ^[a-z0-9][a-z0-9-]{0,35}$ ]] || die "NAME: [a-z0-9][a-z0-9-]*, at most 36 chars"
W=$B/$NAME
SLUG="ariadne-ci-$NAME"

stage() {
  [ -s "$B/opencv_slug" ] || die "run '$0 opencv' once first"
  local commit; commit="$(git rev-parse --verify "${1:-HEAD}^{commit}")"
  rm -rf "$W"; mkdir -p "$W"
  echo "$commit" > "$W/commit"
  # The code under test: tracked files of COMMIT only (a dirty tree never leaks in), plus the commit id it claims.
  local src="$W/src"; mkdir -p "$src"
  git archive --format=tar --add-virtual-file=.kaggle_commit:"$commit" -o "$src/repo.tar.keep" "$commit" -- . "${EXCLUDE[@]}"
  tar tf "$src/repo.tar.keep" | grep -q -x -E 'IDEA.md|docs/ARIADNE_PHASED_PLAN.md' && die "excluded file in repo.tar.keep"
  echo "ariadne-ci-src-${commit:0:12}" > "$W/src_slug"
  dataset_up "ariadne-ci-src-${commit:0:12}" "ci_kaggle/$NAME/src" "ariadne-ci-src-${commit:0:12}"
  # Inputs, content-addressed by their checksum list (paths + sha256), as ci_aws_launch.sh stages them.
  local ins=(); read -r -a ins <<< "$(echo ${INPUTS-$DEFAULT_INPUTS})"
  : > "$W/inputs_slug"
  if [ ${#ins[@]} -gt 0 ]; then
    # INPUTS_ROOT: the checkout the INPUTS paths are relative to (another session's worktree), default this one.
    # find -L / tar h dereference symlinks (a worktree's .cache points into the main checkout).
    local ir="${INPUTS_ROOT:-$REPO_ROOT}" d="$REPO_ROOT/$W/inputs"; mkdir -p "$d"
    local p; for p in "${ins[@]}"; do [ -d "$ir/$p" ] || die "missing input $ir/$p (set INPUTS / INPUTS_ROOT)"; done
    (cd "$ir" && find -L "${ins[@]}" -type f -print0 | sort -z | xargs -0 sha256sum) > "$d/inputs.SHA256SUMS"
    printf '%s\n' "${ins[@]}" > "$d/inputs.list"
    local h; h="$(sha256sum "$d/inputs.SHA256SUMS" | cut -c1-12)"
    if ! kg datasets status "$(user)/ariadne-in-$h" 2>/dev/null | grep -q ready; then
      tar chf "$d/inputs.tar.keep" -C "$ir" "${ins[@]}" -C "$d" inputs.SHA256SUMS inputs.list
      rm "$d/inputs.SHA256SUMS" "$d/inputs.list"
      dataset_up "ariadne-in-$h" "ci_kaggle/$NAME/inputs" "ariadne-in-$h"
    fi
    rm -rf "$d"; echo "ariadne-in-$h" > "$W/inputs_slug"
  fi
  rm -rf "$src"
  log "staged $commit as $NAME: src $(cat "$W/src_slug"), inputs ${ins[*]:-none} ($(cat "$W/inputs_slug"))"
}

submit() {
  [ -s "$W/commit" ] || die "$NAME is not staged (stage $NAME first)"
  local u commit k; u="$(user)"; commit="$(cat "$W/commit")"; k="$W/kernel"
  rm -rf "$k"; mkdir -p "$k"
  # kernel.py: the job's parameters + this working tree's ci_kaggle.sh, embedded; always exits 0 so Kaggle keeps the
  # output of a failed job too (the job's own rc is manifest.yaml's exit).
  python3 - "$k/kernel.py" infra/eval/ci_kaggle.sh <<EOF
import base64, json, sys
out, script = sys.argv[1:3]
env = {"NAME": "$NAME", "COMMIT": "$commit", "JOBS": "${JOBS:-4}", "CTEST_JOBS": "${CTEST_JOBS:-4}"}
env["TESTS"] = base64.b64decode("$(printf '%s' "${TESTS:-}" | base64 -w0)").decode()
env["EXTRA"] = base64.b64decode("$(printf '%s' "${EXTRA:-}" | base64 -w0)").decode()
src = base64.b64encode(open(script, "rb").read()).decode()
open(out, "w").write(f'''# Ariadne off-AWS CI job {env["NAME"]} (generated by infra/eval/ci_kaggle_launch.sh; development evidence)
import base64, os, subprocess
open("/kaggle/working/ci_kaggle.sh", "wb").write(base64.b64decode("{src}"))
env = dict(os.environ, **{json.dumps(env)})
rc = subprocess.call(["bash", "/kaggle/working/ci_kaggle.sh"], env=env)
print("ci_kaggle.sh exit", rc)
''')
EOF
  local ds=("$u/$(cat "$B/opencv_slug")" "$u/$(cat "$W/src_slug")")
  [ -s "$W/inputs_slug" ] && ds+=("$u/$(cat "$W/inputs_slug")")
  python3 - "$k/kernel-metadata.json" "$u/$SLUG" "$SLUG" "${ds[@]}" <<'EOF'
import json, sys
out, kid, title, *ds = sys.argv[1:]
json.dump({"id": kid, "title": title, "code_file": "kernel.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
           "dataset_sources": ds, "competition_sources": [], "kernel_sources": []}, open(out, "w"), indent=2)
EOF
  local msg; msg="$(kg kernels push -p "/b/ci_kaggle/$NAME/kernel" -t "$(( ${MAX_MINUTES:-90} * 60 ))" 2>&1 | tail -1)"
  case "$msg" in
    *"successfully pushed"*) ;;
    *"session count"*) log "all Kaggle CPU sessions busy: $msg"; exit 75 ;;
    *) die "push failed: $msg" ;;
  esac
  local id; id="$(date -u +%Y%m%dT%H%M%SZ)-kaggle-$NAME"
  echo "$id" > "$W/run_id"
  mkdir -p "build/runs-offaws/$id"
  { echo "name: $NAME"; echo "commit: $commit"; echo "kernel: $u/$SLUG"; echo "datasets: [${ds[*]}]"
    echo "tests: \"${TESTS:-<all>}\""; echo "extra: $(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "${EXTRA:-}")"
    echo "pushed_utc: $(date -u +%FT%TZ)"; echo "host_tree: $(git rev-parse HEAD) $(git status --porcelain | wc -l) dirty paths"
  } > "build/runs-offaws/$id/launch.yaml"
  log "pushed $u/$SLUG (commit ${commit:0:12}) -> build/runs-offaws/$id/"
}

wait_pull() {
  [ -s "$W/run_id" ] || die "$NAME was not submitted"
  local u id st i; u="$(user)"; id="$(cat "$W/run_id")"
  for ((i = 0; ; i++)); do
    st="$(kg kernels status "$u/$SLUG" 2>&1 | tail -1 || true)"
    case "$st" in *COMPLETE*|*ERROR*|*CANCEL*) break ;; esac
    (( i % 10 == 0 )) && log "$SLUG: $st"
    sleep 60
  done
  log "$SLUG: $st"
  local o="build/runs-offaws/$id"
  rm -rf "$W/out"; mkdir -p "$W/out"
  kg kernels output "$u/$SLUG" -p "/b/ci_kaggle/$NAME/out" -o > /dev/null 2>&1 || log "output pull failed"
  [ -d "$W/out/results" ] && cp -r "$W/out/results/." "$o/"
  cp "$W/out/"*.log "$o/kernel.log" 2>/dev/null || true
  echo "kernel_status: \"$st\"" >> "$o/launch.yaml"
  [ -f "$o/manifest.yaml" ] || die "no manifest.yaml in the output ($o/kernel.log)"
  grep -E '^(ctest_summary|ctest_skipped_or_not_run|build_rc|ctest_rc|extra_rc|exit|seconds):' "$o/manifest.yaml" >&2
  return "$(sed -n 's/^exit: //p' "$o/manifest.yaml")"
}

case "$1" in
  stage) stage "${3:-}" ;;
  submit) submit ;;
  wait) wait_pull ;;
  run) submit; wait_pull ;;
  state) kg kernels status "$(user)/$SLUG" 2>&1 | tail -1 || true ;;
  teardown) kg kernels delete "$(user)/$SLUG" -y >/dev/null 2>&1 || true; rm -f "$W/run_id"; log "teardown $NAME done" ;;
esac
