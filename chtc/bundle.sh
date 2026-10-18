#!/bin/bash
# Stage everything a CHTC submission needs into one flat directory.
#
#   chtc/bundle.sh [out_dir]      # default: chtc_bundle/
#   scp -r chtc_bundle/ <netid>@<ap>.chtc.wisc.edu:~/sepsis
#
# The submit files reference their inputs by bare name (plus data/), so they
# only work from inside this flat layout, never from the repo root.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-"$repo/chtc_bundle"}

mkdir -p "$out/data" "$out/logs"
cp "$repo"/chtc/*.sub "$repo"/chtc/*.sh "$repo"/chtc/*_runs.txt "$out/"
cp "$repo/causal_discovery.py" "$repo/tiers.py" "$out/"
cp "$repo"/data/*.csv "$out/data/"
rm -f "$out/bundle.sh"
chmod +x "$out/causal_discovery.sh"

echo "Staged $out"
