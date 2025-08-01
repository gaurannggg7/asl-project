#!/usr/bin/env bash
set -euo pipefail

REPO=https://github.com/gaurannggg7/Sign-Language-Mocap-Archive.git
DEST=${1:-asl_media}                 # optional arg → output folder
EXTS=(mp4 gif mkv webm)              # video-like formats

TMP=$(mktemp -d)

# sparse clone (tree only, no LFS blobs)
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse "$REPO" "$TMP" >/dev/null
git -C "$TMP" sparse-checkout set "SG ASL Dictionary/**"

echo "scanning repo…"
find_args=()
for e in "${EXTS[@]}"; do find_args+=( -iname "*.${e}" -o ); done
unset 'find_args[-1]'                # drop trailing -o

files=()
while IFS= read -r -d '' f; do files+=("$f"); done \
  < <(find "$TMP/SG ASL Dictionary" "${find_args[@]}" -print0)

echo "found ${#files[@]} clips"
mkdir -p "$DEST"
for f in "${files[@]}"; do
  rel=${f#"$TMP"/}
  mkdir -p "$DEST/$(dirname "$rel")"
  cp "$f" "$DEST/$rel"
done

echo "done → $(find "$DEST" -type f | wc -l) files copied to $DEST/"
