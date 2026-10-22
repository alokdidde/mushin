#!/usr/bin/env bash
# Clone uzumaki at the commit pinned in UZUMAKI_COMMIT and fetch the MuJoCo assets for every body.
# Usage: scripts/fetch_uzumaki.sh [DIR]   (default: third_party/uzumaki)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dir="${1:-$root/third_party/uzumaki}"
commit="$(cat "$root/UZUMAKI_COMMIT")"
if [ ! -d "$dir/.git" ]; then
  git clone -q https://github.com/alokdidde/uzumaki "$dir"
fi
git -C "$dir" fetch -q origin
git -C "$dir" checkout -q "$commit"
PYTHONPATH="$dir/src" python -m uzumaki.sim.assets fetch so101 panda spot g1 go1 umi aloha tidybot go1_walk
echo "uzumaki $commit ready in $dir"
