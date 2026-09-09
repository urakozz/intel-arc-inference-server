#!/usr/bin/env python3
"""Generate the deterministic T1 tokenizer parity corpus (stdlib only)."""
import os
import random


RNG = random.Random(20260906)
OUT = "tests/tokenizer/corpus.txt"
N = 1138
ADDED_TOKENS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
    "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
    "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
    "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
    "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<tool_response>",
    "</tool_response>", "<think>", "</think>", "<|audio_start|>",
    "<|audio_end|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>",
    "<tts_text_bos_single>", "<|audio_pad|>",
]

WORDS = ["amber", "binary", "careful", "distant", "engine", "forest", "gentle",
         "harbor", "inference", "jagged", "kernel", "lantern", "model", "north",
         "orange", "prompt", "quiet", "river", "signal", "token", "useful", "vector"]
VERBS = ["builds", "carries", "finds", "guides", "keeps", "measures", "moves", "reads"]
CJK = ["今天模型读取新的提示。", "東京で静かな川を見た。", "한국어 토큰을 정확히 읽는다.",
       "汉字とかなを混ぜて書く。", "서울의 작은 엔진은 준비됐다。"]
EMOJI = ["😀", "👩🏽‍💻", "👨‍👩‍👧‍👦", "🏳️‍🌈", "🇩🇪", "🇯🇵", "👍🏿", "🧑🏻‍🚀", "🫶🏼", "🚀"]
CONTRACTIONS = ["it's", "IT'S", "They'VE", "we're", "DON'T", "I'd", "you're", "CAN'T"]
DIGITS = ["2026", "3.14159", "1,000,000", "１２３", "0xB70", "-42", "6.02e23", "٠١٢٣"]


def choose(words, n):
    return " ".join(RNG.choice(words) for _ in range(n))


def english(i):
    return f"The {RNG.choice(WORDS)} {RNG.choice(VERBS)} {choose(WORDS, 5)} case {i}."


def code(i):
    indent = ("", "\t", "    ", "        ")[i % 4]
    forms = [
        f"value_{i} = {{\"token\": {i}, \"ok\": true}}",
        f"if (count_{i} >= 3) {{ return map[{i % 17}]; }}",
        f"std::vector<int> ids_{i} = {{{i % 7}, {i % 11}, {i % 13}}};",
        f"def parse_{i}(text: str) -> dict: return {{'n': len(text)}}",
    ]
    return indent + forms[i % len(forms)] + (" " * (i % 5))


def cjk(i):
    return f"{RNG.choice(CJK)} {''.join(RNG.choice('模型東京서울漢字') for _ in range(3 + i % 5))}"


def emoji(i):
    return f"{RNG.choice(EMOJI)} {RNG.choice(EMOJI)} launch {i % 100} {RNG.choice(EMOJI)}"


def whitespace(i):
    k = i % 40 + 1
    forms = [" " * k, "\t" * k, " " * k + "\t" * (41 - k),
             "\t" * (41 - k) + " " * k, "prefix" + " " * k + "suffix",
             "carriage" + "\r" + "return" + " " * k]
    return forms[i % len(forms)]


def contractions(i):
    c = RNG.choice(CONTRACTIONS)
    return f"{c} {RNG.choice(CONTRACTIONS)}: case-{i}, {c.upper()} and {c.lower()}."


def digits(i):
    return f"record {RNG.choice(DIGITS)} / {RNG.choice(DIGITS)} = {i:04d}"


def added(i):
    token = ADDED_TOKENS[i % len(ADDED_TOKENS)]
    neighbor = ADDED_TOKENS[(i + 1) % len(ADDED_TOKENS)]
    form = i % 3
    if form == 0:
        return token
    if form == 1:
        return f"prefix {token} suffix"
    return token + neighbor


def mixed(i):
    return (f"{RNG.choice(CJK)}\t{RNG.choice(EMOJI)} {RNG.choice(DIGITS)} "
            f"{RNG.choice(CONTRACTIONS)} {choose(WORDS, 3)}")


def main():
    families = {
        "english": [english(i) for i in range(N)],
        "code": [code(i) for i in range(N)],
        "cjk": [cjk(i) for i in range(N)],
        "emoji": [emoji(i) for i in range(N)],
        "whitespace": [whitespace(i) for i in range(N)],
        "contractions": [contractions(i) for i in range(N)],
        "digits": [digits(i) for i in range(N)],
        "added_tokens": [added(i) for i in range(N)],
        "mixed": [mixed(i) for i in range(1136)],
    }
    cases = [case for family in families.values() for case in family]
    assert len(cases) == 10240
    assert all("\n" not in case for case in cases)
    assert all(len(family) >= 1000 for family in families.values())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(cases) + "\n")
    for name, family in families.items():
        print(f"{name}: {len(family)}")
    print(f"total: {len(cases)}")


if __name__ == "__main__":
    main()
