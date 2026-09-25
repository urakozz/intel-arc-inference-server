#!/usr/bin/env python3
"""passkey.py - the spec 6 K3 passkey-retrieval prompt.

    passkey.py <snapshot> <placement 0..1> <out.ids> [n_target]

Filler ("The grass is green. The sky is blue. The sun is yellow. Here we go. There
and back again." repeated) with "The pass key is 71432. Remember it. 71432 is the
pass key." inserted after the fraction <placement> of the filler, ending in "What is
the pass key? The pass key is". Tokenised with the snapshot's tokenizer.json, no
special tokens and no chat template (a base continuation), to about n_target ids
(default 120000). Written one id per line. Prints the id count and the needle's
position in ids.

Run in the oracle container (tools/probe/passkey.sh does). Plan:
docs/superpowers/plans/2026-09-25-spec6b-flash-attn-integration-and-128k.md, Task 5.
"""
import os
import sys

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
NEEDLE = "The pass key is 71432. Remember it. 71432 is the pass key. "
QUESTION = "What is the pass key? The pass key is"


def main() -> None:
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    snap, place, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    target = int(sys.argv[4]) if len(sys.argv) == 5 else 120000
    if not 0.0 <= place <= 1.0:
        sys.exit("placement must be in [0, 1]")
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
        encode = lambda t: tok.encode(t, add_special_tokens=False).ids  # noqa: E731
    except ImportError:
        from transformers import AutoTokenizer
        atok = AutoTokenizer.from_pretrained(snap)
        encode = lambda t: atok.encode(t, add_special_tokens=False)  # noqa: E731
    per = len(encode(FILLER * 100)) / 100.0
    reps = int((target - len(encode(NEEDLE)) - len(encode(QUESTION))) / per)
    before = int(round(place * reps))
    head = FILLER * before
    text = head + NEEDLE + FILLER * (reps - before) + QUESTION
    ids = encode(text)
    needle_at = len(encode(head))
    with open(out, "w") as f:
        f.write("\n".join(str(i) for i in ids) + "\n")
    print(f"passkey: placement {place}, {len(ids)} ids, needle at id {needle_at} "
          f"({100.0 * needle_at / len(ids):.1f} %), -> {out}")


if __name__ == "__main__":
    main()
