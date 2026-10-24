#!/usr/bin/env bash
# Entry point inside a RunPod pod (set as the pod's start command by mushin.runpod.launch).
# Clones mushin at MUSHIN_COMMIT, installs it, fetches inputs from the Hugging Face Hub, runs
# MUSHIN_COMMAND under a hard deadline, uploads MUSHIN_OUT_DIR, and always stops the pod.
set -uo pipefail
: "${MUSHIN_COMMIT:?}" "${MUSHIN_COMMAND:?}" "${MUSHIN_MAX_SECONDS:?}"
OUT_DIR="${MUSHIN_OUT_DIR:-runs}"
EXTRAS="${MUSHIN_EXTRAS:-train}"
WORK=/workspace/mushin
log() { echo "[mushin $(date -u +%H:%M:%S)] $*"; }
# Everything the pod prints also goes to a file that finish() uploads, so a job that fails
# before it writes any output still leaves its log on the Hub.
LOG=/workspace/pod_job.log
mkdir -p /workspace && exec > >(tee -a "$LOG") 2>&1

hub_upload() {  # hub_upload LOCAL_PATH PATH_IN_REPO [extra hf upload args]
  local local_path=$1 repo_path=$2; shift 2
  local prefix="${MUSHIN_OUT_PREFIX:-}"
  [ -n "$prefix" ] && [ -n "$repo_path" ] && prefix="$prefix/"
  timeout 1800 hf upload "$MUSHIN_OUT_REPO" "$local_path" "$prefix$repo_path" \
    --repo-type "${MUSHIN_OUT_REPO_TYPE:-model}" --private "$@" >/dev/null 2>&1
}

# Shards (shards/sNNNNN/ with a DONE marker) go up as soon as they are written, so a pod that
# dies loses at most the shard in progress. Each shard's tar is removed once it is uploaded.
upload_shards() {
  local shard
  for shard in "$WORK/$OUT_DIR"/shards/*/; do
    [ -f "$shard/DONE" ] && [ ! -f "$shard/UPLOADED" ] || continue
    if hub_upload "$shard" "shards/$(basename "$shard")" --exclude "DONE"; then
      touch "$shard/UPLOADED"; rm -f "$shard/images.tar"
    else
      log "shard upload failed: $(basename "$shard")"
    fi
  done
}

finish() {
  status=$?
  log "job exit status $status"
  [ -n "${HEARTBEAT:-}" ] && kill "$HEARTBEAT" 2>/dev/null
  if [ -n "${MUSHIN_OUT_REPO:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
    mkdir -p "$WORK/$OUT_DIR" && cp "$LOG" "$WORK/$OUT_DIR/pod_job.log"
    echo "{\"status\": $status, \"pod\": \"${RUNPOD_POD_ID:-}\"}" > "$WORK/$OUT_DIR/pod_status.json"
    upload_shards
    # Loose frames stay out: the Hub caps a folder at 10k files, and the shards carry them.
    local exclude=()
    [ -d "$WORK/$OUT_DIR/shards" ] && exclude=(--exclude "images/*" --exclude "shards/*")
    for attempt in 1 2 3; do
      hub_upload "$WORK/$OUT_DIR" "" "${exclude[@]}" && break
      log "upload attempt $attempt failed"; sleep 30
    done
  fi
  log "stopping pod"
  if command -v runpodctl >/dev/null; then
    runpodctl stop pod "${RUNPOD_POD_ID:-}" && return
  fi
  curl -fsS -X POST -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
    "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID:-}/stop" || log "could not stop pod"
}
trap finish EXIT

url="https://github.com/alokdidde/mushin"
[ -n "${GITHUB_TOKEN:-}" ] && url="https://x-access-token:${GITHUB_TOKEN}@github.com/alokdidde/mushin"
git clone -q "$url" "$WORK" && git -C "$WORK" checkout -q "$MUSHIN_COMMIT" || exit 10
cd "$WORK"
# Python 3.12 through uv, whatever the image ships.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh || exit 11
export PATH="$HOME/.local/bin:$PATH"
uv venv -q -p 3.12 /workspace/venv && . /workspace/venv/bin/activate || exit 11
uv pip install -q -e ".[$EXTRAS]" "huggingface_hub[cli]>=0.30" || exit 11
if [ "$EXTRAS" != "${EXTRAS/sim/}" ]; then
  (apt-get update -qq && apt-get install -y -qq libosmesa6 libegl1 git >/dev/null) || exit 12
  scripts/fetch_uzumaki.sh || exit 13
fi
if [ -n "${MUSHIN_DATA_REPO:-}" ]; then
  hf download "$MUSHIN_DATA_REPO" --repo-type dataset --local-dir "${MUSHIN_DATA_DIR:-data/m4}" \
    || exit 14
fi
if [ -n "${MUSHIN_RESUME:-}" ] && [ -n "${MUSHIN_OUT_REPO:-}" ]; then
  # Resume after a pod died: fetch what earlier pods uploaded (shard records, checkpoints; not
  # the frame tars, which are already on the Hub) into OUT_DIR and let the job continue from it.
  prefix="${MUSHIN_OUT_PREFIX:+$MUSHIN_OUT_PREFIX/}"
  if hf download "$MUSHIN_OUT_REPO" --repo-type "${MUSHIN_OUT_REPO_TYPE:-model}" \
       --include "${prefix}*" --exclude "*.tar" --local-dir "$WORK/.resume" >/dev/null 2>&1 \
     && [ -d "$WORK/.resume/$prefix" ]; then
    mkdir -p "$WORK/$OUT_DIR" && cp -r --update=none "$WORK/.resume/$prefix". "$WORK/$OUT_DIR/"
    for shard in "$WORK/$OUT_DIR"/shards/*/; do
      [ -d "$shard" ] && touch "$shard/DONE" "$shard/UPLOADED"
    done
    log "resumed from $MUSHIN_OUT_REPO/$prefix"
  else
    log "nothing to resume"
  fi
fi
if [ -n "${MUSHIN_OUT_REPO:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
  # Heartbeat: finished shards go up every 5 min, and progress.json (if the job writes one)
  # every 30 min.
  (beat=0; while sleep 300; do
     upload_shards
     beat=$((beat + 1))
     [ $((beat % 6)) -eq 0 ] && [ -f "$WORK/$OUT_DIR/progress.json" ] \
       && hub_upload "$WORK/$OUT_DIR/progress.json" progress.json
     # MUSHIN_SYNC_OUT: training stages push their checkpoints every 30 min, for MUSHIN_RESUME.
     [ $((beat % 6)) -eq 0 ] && [ -n "${MUSHIN_SYNC_OUT:-}" ] && [ -d "$WORK/$OUT_DIR" ] \
       && hub_upload "$WORK/$OUT_DIR" "" --exclude "images/*" --exclude "shards/*"
   done) &
  HEARTBEAT=$!
fi
log "running: $MUSHIN_COMMAND"
timeout "$MUSHIN_MAX_SECONDS" bash -c "$MUSHIN_COMMAND"
