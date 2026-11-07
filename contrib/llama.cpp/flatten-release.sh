#!/bin/sh
# flatten-release.sh <worktree of branch flattened> <release tag> <pin tarball>
#
# One commit on the branch `flattened` for a release: the pinned llama.cpp with
# that release's patch series applied, so the code can be read and searched as
# files instead of as diffs. The branch holds the pristine pin as its first
# commit and one commit per release after it, so `git diff <first> HEAD` is what
# arcint changes and the diff between two commits is what a release changed.
# Nothing builds from this branch: the recipe stays pin + contrib/llama.cpp/patches.
#
# The first call on an empty orphan worktree writes the pin commit. The tree must
# come out as pin + series byte for byte, apart from the two exceptions below.
#   - .github/ is left out: llama.cpp's workflows would run in this repository.
#   - FLATTENED.md is added, the page that says what the branch is.
set -eu
W=$1; TAG=$2; PIN=$3
REPO=$(git -C "$W" rev-parse --git-common-dir)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

mkdir "$T/pin" "$T/series"
tar -xzf "$PIN" -C "$T/pin" --strip-components=1
git --git-dir="$REPO" archive "$TAG" contrib/llama.cpp/patches contrib/llama.cpp/README.md | tar -x -C "$T/series"

sync_tree() {   # replace the worktree's files with $1, keeping .git
    find "$W" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
    (cd "$1" && tar -cf - --exclude=./.github .) | tar -xf - -C "$W"
}

if ! git -C "$W" rev-parse -q --verify HEAD >/dev/null; then
    sync_tree "$T/pin"
    git -C "$W" add -A -f
    git -C "$W" commit -q -m "llama.cpp at the pin of arcint $TAG, unmodified, without .github/" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
fi

cd "$T/pin"
git init -q .
for p in "$T"/series/contrib/llama.cpp/patches/*.patch; do
    git apply --whitespace=nowarn "$p" || { echo "does not apply: $p" >&2; exit 1; }
done
rm -rf .git
n=$(ls "$T"/series/contrib/llama.cpp/patches/*.patch | wc -l)
sync_tree "$T/pin"
cat > "$W/FLATTENED.md" <<EOF
# arcint: the patched llama.cpp, flattened

This branch of [arcint](https://github.com/marfrit/arcint) holds llama.cpp at the
commit arcint pins, with arcint's patch series applied, so the code can be read
and searched as files. This is $TAG: $n patches.

- The first commit is the unmodified pin; each later commit is one release. The
  diff between the first commit and any other is what arcint changes in llama.cpp.
- Most of the work is the OpenCL backend for Intel Arc:
  \`ggml/src/ggml-opencl/ggml-opencl.cpp\` and \`ggml/src/ggml-opencl/kernels/\`.
- Nothing builds from this branch. arcint builds from the pin plus
  \`contrib/llama.cpp/patches\` on \`main\`, and \`contrib/llama.cpp/README.md\` there
  has each patch's measurements.
- \`.github/\` is left out of this tree, so that llama.cpp's workflows do not run here.
- llama.cpp is MIT licensed; see \`LICENSE\`. The patches carry arcint's license.
EOF
git -C "$W" add -A -f
git -C "$W" commit -q -m "arcint $TAG: the series applied, $n patches" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git -C "$W" log --oneline -2
