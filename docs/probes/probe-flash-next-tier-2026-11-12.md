# Flash-Next expert tier: the hit rate and the link rates (2026-11-12)

Two numbers decide the two-card design ([spec 21 v2](../superpowers/specs/2026-11-11-spec21v2-flash-next-v2-design.md)
sections 3 and 5): **h**, the share of routed expert reads served from VRAM when the most-used experts are resident, and
**P**, the rate at which a card pulls the rest from pinned host RAM. This is plan 21v2a Task 4 (hit rate) and Task 5
(links), run before our own quantisation exists (it starts Saturday 2026-11-14).

Branch `flash-next-probes`. Nothing in the engine changed; the one code change is a new `peer` mode in
`tools/probe/probe_offload.cc` (the plan's Task 5 Step 2).

## 1. Hit rate h (Task 4): from Intel's checkpoint, not ours

**What ran.** No GPU and no `--layers N` run was needed: the 2026-11-09 routing traces already hold the routes of **all 48
layers** (plus the MTP head), recorded teacher-forced by the CPU reference on **Intel's** `W4A16-AutoRound` checkpoint. 39
sources, 80,604 token rows, 38.7 million routed slots (10 of 512 experts a layer): the 36 tool-call scenarios
(`t1`..`t6` x 6 files, 960-3,000 rows each), one agentic transcript (12,253 rows), and a short `code` (61 rows) and
`prose` (42 rows) text. `tools/oracle/offload_curve.py counts|curve|mtp` ran over them in the oracle image on the box CPU
(`tools/oracle/run_in_container.sh`, 8 threads, 24 GB cap), about 10 minutes. The routers are bf16 in both checkpoints, so
the routes are expected to be close to ours, but **ours is not measured**: re-record the 39 sources on our export and rerun
this (plan 21v2a Task 4 Step 4) once it exists.

**Placement and hold-out.** Static "most-used first": per layer, the f x 512 experts with the highest counts in the profile
are resident (the "uniform" column: the same share in every layer; "per layer": a global greedy that gives layers with a
flatter routing more slots). The sources are split in two halves by sha256 of the name (the tool-call scenarios 18 / 18);
the profile is built on one half and scored on the other, both ways, and the two scores are averaged ("held out"). h is the
share of all routed slots that hit.

| resident fraction f | uniform, held out | per layer, held out | tool-call scenarios (held) | agentic transcript (held) | code (61 rows, held) | prose (42 rows, held) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.839 | 0.844 | 0.851 | 0.803 | 0.803 | 0.430 |
| 0.70 | 0.945 | 0.950 | 0.953 | 0.932 | 0.916 | 0.638 |
| 0.75 | 0.962 | 0.966 | 0.968 | 0.953 | 0.938 | 0.705 |
| **0.80** | **0.976** | **0.979** | 0.980 | 0.969 | 0.954 | 0.754 |
| 0.85 | 0.986 | 0.988 | 0.989 | 0.982 | 0.968 | 0.830 |
| 0.90 | 0.994 | 0.995 | 0.995 | 0.992 | 0.985 | 0.891 |

In sample (profile and score on the same traces) f = 0.8 gives 0.980 / 0.983, so the profile generalises: held out loses
0.4 points. f to reach h = 0.90 / 0.95 / 0.99 held out: 0.60 / 0.72 / 0.88 (uniform). The full table (f 0.30 to 1.00) is
in `curve.txt` on the box, `~/b70/trees/flash-next-probes/oracle-out-fnp-curve/`.

**Reading it.**
- The spec's estimate (h about 0.96 at f = 0.80, from basalt's f^(1/6) rule) is **conservative**: the measured held-out h
  is **0.976-0.979** at f = 0.80, and the f^(1/6) rule would give 0.964 only at f = 0.80 while here 0.50 already gives
  0.84 against the rule's 0.89, so the real curve is flatter at the low end and steeper near the top. Routing in these
  traces is concentrated: a few experts per layer take most of the traffic.
- The worst held-out class at f = 0.80 is code at 0.954 (61 rows only) and the long agentic transcript at 0.969; both are
  scored by a profile built on the *other* sources, so the figure is a cross-domain one. Prose (42 rows) is too short to
  mean anything (it touches 75-178 experts a layer; at 42 tokens nothing is concentrated).
- **Caveats, in order of weight.** (1) The traces are teacher-forced routes over whole prompts and answers: the prompt
  tokens dominate (an A4 file is a 900-2,000 token prompt plus a short answer). Routes of free-running decode tokens may
  concentrate differently; not measured here. (2) Intel's checkpoint, not ours. (3) One agentic transcript, 12k rows, is the
  only long-session evidence; the profile for the real workload should come from a few such sessions. (4) A static
  placement: no swap policy was simulated (`offload_curve.py swap` was not run; its inputs include the copy rate measured
  below, but the plan treats swaps as a later step). (5) The cross-check "which layers does this cover" is all 48; the
  engine itself was not run, so nothing here depends on the 18 / 37 layers that fit.
- The MTP head's own routes: 486 of 512 experts routed; if cached at f = 0.8 the head would hit 0.989.
- A multi-token verify step (MTP K = 1..4) touches more distinct experts a layer: 17.2 / 23.5 / 29.2 / 34.5 of 512 for
  K = 1..4 (M = K + 1 rows), against 10 for a plain step; the misses a verify step, over all 48 layers, are 19.1 / 27.8 /
  36.3 / 44.7 experts at f = 0.8 (uniform, in sample), against about 10 for a plain step (480 slots at h = 0.98). A verify
  step therefore reads roughly 2-4.5 times a plain step's misses.

## 2. Link rates (Task 5)

**Tool.** `tools/probe/probe_offload.cc` already had the host side (`copy`, `zc`, `hostbw`, `pin`, `alias`, `replay`,
`hold`, from plan 22a). Added: **`peer`** mode (both cards): `po_read_sum` (a kernel that reads in place) on card 0 / card
1 from its own VRAM, from pinned host USM and from the other card's VRAM over P2P, each alone and both cards at once; the
copy engine moving the same bytes card 1 -> card 0, card 0 -> card 1 and host -> card 0, on the compute ordinal and the
copy-only group. Arms are interleaved and timed as a median of 9 after 3 warm-ups; kernel times are device timestamps. Host
allocations: two 1 GiB pinned ranges, 2 GiB in all (under the 4 GiB limit). `zc` (the in-kernel mixed launch with the
real MoE binaries), `pin`, `alias`, `replay`, `hostbw` and `links` were **not** run: they are the plan's other Task 5
stages and need the long quiet session.

**Commands** (on the box, `~/b70/trees/flash-next-probes`, logs `~/b70/logs/flash-next-probes/`):
`tools/gpu_lock.sh card 0 -- build/tools/probe/probe_offload copy --dev 0 --reps 7` and
`tools/gpu_lock.sh box -- build/tools/probe/probe_offload peer --reps 9`.

**What else ran on the box.** The `copy` run (18:05) held card 0 while another session held card 1 (a GPU job of the
Ornith MoE work). The `peer` run (18:08-18:10) held the whole-box lock, so no other GPU job ran; CPU jobs did (an Ornith
CPU reference, load average 23-36 on 44 threads), which can only have lowered the host numbers. Not a record-grade quiet
session: treat the host rates as a floor.

### Host RAM to a card (pinned memory, PCIe 3.0, card 0)

| record | copy engine, copy-only queue, replayed list | copy engine, compute queue | kernel reading host memory in place |
|---|---:|---:|---:|
| 1 expert, 2.61 MB (215-233 us) | 12.1 GB/s | 11.8 | 11.2 |
| 1 expert, down only, 0.87 MB | 11.1 | 10.3 | - |
| 10 experts (a token's routed set, 26 MB) | 10.3-11.0 | 10.5-12.0 | 10.4-12.0 (scattered or contiguous) |
| 64 experts, 167 MB | 11.1-11.4 | 10.8-10.9 | 11.0 |
| 256 experts, 668 MB | 11.3-12.6 | 11.0-12.3 | 12.5 |
| 1 GiB | 11.3-12.0 | 10.8 | 11.8 |

- A 4-byte copy takes 7-8 us (the fixed cost); a whole expert about 215 us on the copy queue.
- **Both cards reading host memory at once** (the tensor-parallel case): 21.3-24.7 GB/s aggregate, i.e. 10.7-12.3 each.
  The two links do not slow each other, and host memory is not the limit.
- For comparison the same kernel reads **its own VRAM** at 560-604 GB/s (26 MB to 1 GiB; 401 at one expert, where launch
  ramp dominates).
- **P to use: 11.2 GB/s** (one expert in place, the size a miss has) and 12.0 GB/s (10 experts) bracket it; the spec
  assumed 12. In-kernel reads are as fast as the copy engine, so zero-copy reads of misses cost no extra staging.

### Card to card over P2P (cards 0 and 1)

| operation | rate | note |
|---|---:|---|
| card 0 kernel **reads** card 1's VRAM | **1.80 GB/s** | flat from 2.6 MB to 1 GiB (1.45 ms for one expert) |
| card 1 kernel reads card 0's VRAM | 1.80 GB/s | same |
| both read each other at once | 3.6 GB/s aggregate | 1.8 each: no sharing penalty |
| copy engine **pulling** (list on the destination card, card 0 copies card 1 -> card 0) | 1.80 GB/s | either queue group |
| copy engine **pushing** (list on the source card: card 1 -> card 0 listed on card 1; card 0 -> card 1 listed on card 0) | **9.4-9.9 GB/s** | 0.27 ms for one expert, 9.9 at 64 experts and up |

Pulling over P2P (a card reading the other's memory, by kernel or by copy engine) runs at 1.8 GB/s, six times below the
host link; **pushing** from the owning card runs at 9.9 GB/s, 10-15 % below the host link. The cards' own P2P reads are
therefore not a usable expert tier; a pushed copy would cost about as much as a host read.

## 3. Decode speed from the two formulas

Spec 21 v2 section 5, with its constants (457.8 GB/s VRAM, D = 3.76 GB and R = 1.253 GB for `pp`; 2.53 and 0.627 per card
plus 0.34 ms of all-reduces for `tp`), the measured h and P, serial law (VRAM part and miss part add: the plan's upper
bound on cost; `zc`'s mixed-launch law was not measured, an overlap would be faster). MTP off, one request, short context.

| case | h | P | (a) `pp` + tier | (b) `tp` + tier |
|---|---:|---:|---:|---:|
| measured, uniform f = 0.80 | 0.976 | 11.2 | **73.6 t/s** | **116.9 t/s** |
| measured, uniform f = 0.80 | 0.976 | 12.0 | 74.6 | 118.1 |
| per-layer fill f = 0.80 | 0.979 | 11.2 | 75.2 | 118.9 |
| worst class (code), f = 0.80 | 0.954 | 11.2 | 62.6 | 102.5 |
| long agentic transcript, f = 0.80 | 0.969 | 11.2 | 69.6 | 111.7 |
| f = 0.75 (more KV or MTP in VRAM) | 0.962 | 11.2 | 66.3 | 107.5 |
| f = 0.70 | 0.945 | 11.2 | 59.0 | 97.7 |
| f = 0.85 | 0.986 | 11.2 | 80.1 | 124.9 |
| the spec's pessimistic case | 0.85 | 11.2 | 36.6 | 64.8 |
| everything in VRAM | 1.00 | - | 91.3 | 138.2 |

The formulas as written give 91 / 138 t/s at h = 1.0, where the spec's own table says 97 / 145, and 71 against 75 at
h = 0.97: the spec's table is about 5 % higher than its own formula with 457.8 GB/s; the figures above are the formulas
evaluated here. They are not measured speeds: no Flash-Next launch overhead, no attention at depth, no MTP.

## 4. What this says

- **The tier is cheap enough.** With the measured h (0.976 held out at f = 0.80) the host link costs 2.7 ms of a 13.6 ms
  token on `pp` (20 %) and 1.3 ms of 8.6 ms on `tp` (16 %). h would have to fall to about 0.85 (the spec's pessimistic
  case) to take `pp` below 40 t/s. The gates of spec 21 v2 section 11 (`pp` 30 t/s, `tp` 50 t/s) have a wide margin on the
  traces measured; the open question is whether free-running decode on our own weights keeps h near 0.97.
- **`tp` beats `pp` by about 1.6x** at every h (117 against 74 t/s; 98 against 59 at f = 0.70), because it reads half the
  bytes on each card in parallel and both host links stream misses at once (measured: 21-25 GB/s aggregate with no
  contention). The cost of building `pp` first is that its decode ceiling on these numbers is 75-80 t/s; the spec's reason
  for `pp` first (it is built, ours by an operator ruling) stands, but the speed target "past Strata's 66 t/s" is met by
  `pp` only at h of 0.96 or better, i.e. f of 0.75 or more, while `tp` meets it with room to spare at h = 0.945.
- **A peer expert tier (spec section 5 (c)) is out** on this hardware: P2P reads are 1.8 GB/s (6x below host), and even a
  pushed copy (9.9 GB/s) is slower than the host link.
- Budget lever: each GB of KV or scratch moves f by about 0.015 (13 GB of experts outside VRAM is f = 0.80). At f = 0.70
  (about 5.5 GB less room) `pp` drops to 59 t/s and `tp` to 98.
- Still open for the quiet session: the mixed-launch law (does a miss overlap with VRAM work?), the P2P all-reduce cost
  (0.34 ms is the spec's figure, not measured here), the pinned-range limits (`pin`), and the hit rate on our own export.
