#!/usr/bin/env bash
# tools/box_validate/changed_since.sh SHA - the files changed between SHA and this checkout (committed and not) that
# a validation run could see: everything but the documentation (docs/, *.md). Prints one path a line; nothing when
# only documentation changed. Exit 2 when SHA is not a commit here. Used by box_validate.sh --nightly.
set -u
sha=${1:?usage: changed_since.sh SHA}
git rev-parse --verify --quiet "$sha^{commit}" > /dev/null || { echo "changed_since: $sha is not a commit here" >&2; exit 2; }
{ git diff --name-only "$sha" -- . ':(exclude)docs' ':(exclude)*.md'
  git ls-files --others --exclude-standard -- . ':(exclude)docs' ':(exclude)*.md'; } | sort -u
