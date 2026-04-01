#!/bin/bash
# Move episode_XXXXXXX.npz files into subdirectories of ~1000 files each.
# e.g. episode_0053819.npz -> ep_0053/episode_0053819.npz
#
# Usage: bash shard_calvin_episodes.sh /path/to/training

set -euo pipefail

DIR="${1:?Usage: bash shard_calvin_episodes.sh /path/to/split_dir}"

# Pre-create all shard dirs
seq -w 0000 1999 | xargs -I{} mkdir -p "$DIR/ep_{}"

# Move files in bulk, 1000 at a time
find "$DIR" -maxdepth 1 -name 'episode_*.npz' -print0 | \
    xargs -0 -P 8 -n 500 bash -c '
        for f in "$@"; do
            base=$(basename "$f")
            shard="${base:8:4}"
            mv "$f" "$(dirname "$f")/ep_${shard}/$base"
        done
    ' _

echo "Done sharding in $DIR"
