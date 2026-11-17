#!/usr/bin/env python3
"""The f32 judge on whole served streams (plan 21-QFN-v2a Task 7: the tool-call test on Qwen3.8-Flash-Next, whose bf16 CPU
reference of the set takes days on the box CPU): every graded row of each request, teacher-forced on the stream itself.

  stream_judge.py --dir D --label L --dump DIR --judge DIR [--tag a4_q4exp_int8] [--snapshot DIR] [--vocab-used 248077]

D holds serve_client.py records (requests.jsonl / *.jsonl); the requests of arm L, its lowest round, are judged. Each
request is one forced context: its prompt ids and its output ids up to and including the first end-of-turn id
(serve_judge.py's turn_end: the rows after it are not graded), key <tag>_<set>_<name>_<FNV-1a 64> as
golden::ForcedJudge::key, dumped to --dump (<key>.prompt.ids / .force.ids: what tools/oracle/qwen4exp_quality.py forced
turns into <key>.golden.safetensors in --judge). Row j of the reference is the f32 decision after the prompt and the
first j output ids; the output id there is accepted by golden_common.h judge_row's one-bf16-ulp rule or rejected with its
gap below the f32 top in bf16 ulp. A request PASSES when the judge rejects none of its ids or every rejected id is a
near-miss (<= 8 bf16 ulp, golden::kNearMissUlp); a miss beyond 8 ulp FAILS (the strict bar: there is no second arm to
count against, and b70-serve returns no logits row for the bf16-unstable rule). A request without its reference is
AWAITING. Prints one line per request and the totals; exit 1 when anything FAILS, else 77 when anything awaits, else 0.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import serve_client  # noqa: E402
from serve_judge import NEAR_MISS_ULP, VOCAB_USED, Judge, dump, end_of_turn_ids, forced_key, misses, prompt_of, \
    turn_end  # noqa: E402


def run(a):
    recs = [r for r in serve_client.records(a.dir) if r["label"] == a.label]
    if not recs:
        print(f"stream_judge: no requests of arm '{a.label}' under {a.dir}")
        return 1
    rnd = min(r["round"] or 0 for r in recs)
    recs = sorted((r for r in recs if (r["round"] or 0) == rnd), key=lambda r: (r["set"], r["name"]))
    a.stops, src = end_of_turn_ids(a)
    print(f"stream_judge: arm {a.label} r{rnd}, {len(recs)} requests; end of turn ids {sorted(a.stops)} ({src})")
    c = {"PASS": 0, "FAIL": 0, "AWAIT": 0}
    rows = rej = near = 0
    for r in recs:
        prompt = prompt_of(r, a.root)
        if prompt is None:
            print(f"FAIL {r['set']}/{r['name']}: no prompt ids recorded")
            c["FAIL"] += 1
            continue
        force = list(r["out_ids"][:turn_end(r["out_ids"], a.stops)])
        k = forced_key(f"{a.tag}_{r['set']}_{r['name']}", prompt, force)
        dump(a.dump, k, prompt, force)
        path = os.path.join(a.judge, k + ".golden.safetensors")
        if not os.path.exists(path):
            print(f"AWAIT {r['set']}/{r['name']}: {k}")
            c["AWAIT"] += 1
            continue
        j = Judge(path, force)
        m, _ = misses(j, force, min(a.vocab_used, j.V), len(force))
        worst = max((x[3] for x in m), default=0.0)
        v = "PASS" if worst <= NEAR_MISS_ULP else "FAIL"
        c[v] += 1
        rows += len(force)
        rej += len(m)
        near += sum(1 for x in m if x[3] <= NEAR_MISS_ULP)
        detail = "; ".join(f"row {row}: id {fid} vs f32 top {top}, {g:.2f} ulp" for row, top, fid, g in m)
        print(f"{v} {r['set']}/{r['name']}: {len(force)} graded ids, {len(m)} rejected (worst {worst:.2f} ulp)"
              f"{': ' + detail if m else ''}")
    print(f"stream_judge: {c['PASS']} PASS, {c['FAIL']} FAIL, {c['AWAIT']} awaiting their f32 reference; {rows} graded "
          f"ids, {rej} rejected by the f32 judge ({near} of them near-misses <= {NEAR_MISS_ULP:g} ulp)")
    return 1 if c["FAIL"] else (77 if c["AWAIT"] else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="serve_client.py records")
    ap.add_argument("--label", required=True, help="the arm judged")
    ap.add_argument("--dump", required=True, help="where the forced contexts are written")
    ap.add_argument("--judge", required=True, help="the f32 references (<key>.golden.safetensors)")
    ap.add_argument("--tag", default="a4_q4exp_int8", help="the key's test tag")
    ap.add_argument("--snapshot", default="", help="the model's snapshot (generation_config.json's eos list)")
    ap.add_argument("--vocab-used", type=int, default=VOCAB_USED)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(HERE)))
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
