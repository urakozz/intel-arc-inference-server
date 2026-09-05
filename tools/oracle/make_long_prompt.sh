#!/usr/bin/env bash
# tools/oracle/make_long_prompt.sh -- build the >= 2048-id long prompt
# (`tests/golden/prompts/long.ids` and its readable twin `long.txt`) from the
# three COMMITTED prompts, deterministically, on the Mac. No model, no
# tokenizer, no container: it is `cat` and `awk`.
#
# Spec 2 §6.2 needs one prompt of >= 2048 ids so that a C = 1024 gate crosses a
# chunk boundary in attention-over-cache AND in the GDN state carry. The three
# short prompts are 42 / 61 / 38 = 141 ids, so **twenty repetitions of the trio
# is 2820 ids** -- comfortably over the bar, with the boundary landing inside a
# repetition rather than on one.
#
# **This builds the ID stream, not the text, and that is deliberate.** Plan 6b
# Task 13 Step 1 wrote this script to concatenate the three TEXTS and then
# tokenize the result in the reference container. Two reasons it does not:
#
#   1. `dump.py --prompt` takes an **ids file**, not text (`dump.py:220`), and
#      so does `b70-decode --ids`. The text was only ever an intermediate.
#   2. The tokenizer lives inside the oracle container and nowhere else on the
#      box, and the L1-engine stage is operationally barred from starting
#      containers. Building the ids from the committed `.ids` files removes the
#      dependency entirely and makes the artifact reproducible from a checkout
#      with no model at all.
#
# The consequence, stated rather than glossed: **`long.ids` is the
# CONCATENATION of the three id streams, which is not the same thing as the
# tokenization of the concatenated text** -- a real tokenizer would merge
# differently across each seam. That is irrelevant to what the artifact is for:
# the gate compares the engine against the oracle on the SAME ids, and both
# sides read this file. `long.txt` is written beside it as the decoded-by-hand
# reading of what those ids spell, for review only; nothing consumes it.
set -euo pipefail
cd "$(dirname "$0")/../.."
reps="${REPS:-20}"
ids=tests/golden/prompts/long.ids
txt=tests/golden/prompts/long.txt

: > "$ids"
: > "$txt"
for k in $(seq 1 "$reps"); do
  for p in prose code cjk; do
    tr -s '[:space:]' ' ' < "tests/golden/prompts/$p.ids" >> "$ids"
    printf '\n' >> "$ids"
    printf '=== repetition %d, %s ===\n' "$k" "$p" >> "$txt"
    cat "tests/golden/prompts/$p.txt" >> "$txt"
    printf '\n\n' >> "$txt"
  done
done
n="$(wc -w < "$ids" | tr -d ' ')"
printf '%s: %s ids (%d repetitions of prose+code+cjk)\n' "$ids" "$n" "$reps"
if [ "$n" -lt 2048 ]; then
  echo "make_long_prompt.sh: $n ids is under spec 2 §6.2's 2048 -- raise REPS" >&2
  exit 1
fi
