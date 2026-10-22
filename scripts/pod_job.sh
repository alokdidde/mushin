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

finish() {
  status=$?
  log "job exit status $status"
  if [ -n "${MUSHIN_OUT_REPO:-}" ] && [ -n "${HF_TOKEN:-}" ] && [ -d "$WORK/$OUT_DIR" ]; then
    echo "{\"status\": $status, \"pod\": \"${RUNPOD_POD_ID:-}\"}" > "$WORK/$OUT_DIR/pod_status.json"
    timeout 1800 hf upload "$MUSHIN_OUT_REPO" "$WORK/$OUT_DIR" "${MUSHIN_OUT_PREFIX:-}" \
      --repo-type "${MUSHIN_OUT_REPO_TYPE:-model}" --private || log "upload failed"
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
log "running: $MUSHIN_COMMAND"
timeout "$MUSHIN_MAX_SECONDS" bash -c "$MUSHIN_COMMAND"
