# Spec 16 - pipeline parallel over two B70s

**Status:** design, 2026-10-05, for operator review. Open decisions are marked **(decide)**.

**Scope, set by the operator (2026-10-05):** multi-GPU is **pipeline parallel only** for now; tensor
parallel is not part of this spec.

**Hardware:** the box has two Arc Pro B70s (device 0 at PCI 04:00.0, device 1 at 08:00.0, both on
the single-socket Dell T5810's CPU root complex, PCIe 3.0; a kernel patch enables peer-to-peer, per
the operator). The second card is unplugged while the operator travels; this spec is written now
and probed when both are back. Device 1 is measured 3.2-3.4 % slower than device 0 on prefill and
0.2-0.4 % on decode (docs/10-the-box.md).

Every number is **measured** unless marked **derived** or **estimated**.

## 1. Why, and what it does not do

Pipeline parallel (PP) splits the model's layers between the two cards: layers `[0, s)` on device 0,
`[s, 64)` on device 1 (Qwen3.8), and passes the residual stream from one to the other.

| | PP = 2 (estimates) |
|---|---|
| **single-stream decode** | **not faster**: the layers still run one after the other; the hand-off adds one ~10 KB transfer per token (5120 bf16), a few microseconds over PCIe |
| **memory** | **per card about halved**: ~9 GB of weights each instead of 18 (derived); room for Qwen3.8 at 262144 positions with bf16 KV (17.2 GB of KV split in two), Agnes 3.0 Flash at 128k without spec 12's int8 KV, larger models later |
| **long prefill** | **up to ~2x**: chunk `c` runs layers `[s, 64)` on device 1 while chunk `c + 1` runs `[0, s)` on device 0. The hand-off is ~20 MB per 2048-row chunk (2048 x 5120 bf16), ~1.7 ms at ~12 GB/s, against ~1 s of compute per chunk (derived). This is the agentic case that matters: a prefix-cache miss on a long history |
| **concurrent requests** | ~2x aggregate once spec 13's batching exists (two micro-batches in flight); out of scope here |

## 2. The decisions

**1. One process, one Level Zero context holding both devices.** Not two processes with IPC (as
the TP all-reduce work needed for a multi-rank framework): our engine owns both devices, so
peer access, cross-device copies and events are available inside one context without exporting
handles.

**2. The split point `s` (decide).** Balance memory and time. The embedding (2.54 GB bf16) sits on
device 0, `lm_head` (2.54 GB bf16, 1.27 GB int8) and the MTP head (0.85 GB) on device 1; the FA
layers (the KV holders) are every fourth layer, so any split near the middle gives each card 8 FA
layers. Proposed: `s = 32` (16 FA layers split 8 / 8), revisited by P0 with device 1's 3 % slower
prefill (a split of 33 / 31 may balance time better).

**3. The hand-off mechanism is chosen by P0** among:
- (a) a copy-engine copy of the residual row(s) from device 0 to a buffer on device 1, appended to
  device 0's captured list, signalling an event that device 1's captured list waits on (both lists
  captured once, no host round trip per token);
- (b) a peer write: device 0's last kernel writes the residual straight into device 1's buffer and
  raises a flag; device 1's first kernel polls it;
- (c) host-orchestrated (submit list 0, wait, submit list 1): the baseline.

**Two rules from peer-to-peer work on this exact card pair, applied to (b):** a flag written by the
peer over PCIe must be polled with **system-scope atomics** (a plain or `volatile` load spins on a
stale cache line and can miss the write indefinitely); and the buffer the peer writes into is an
**allocation of its own, page-aligned**, holding nothing the local device writes (peer writes over
PCIe have been seen to disturb neighbouring data in the same allocation). Any device-side sequence
counter lives in device memory and is advanced by the kernel, so a replayed list keeps making
progress with frozen arguments.

## 3. Design

### 3.1 Placement

- Device 0: embedding, layers `[0, s)` with their GDN state, conv ring and KV, decode list 0,
  prefill context 0.
- Device 1: layers `[s, L)` with their state, final norm, `lm_head`, argmax, the MTP head (spec 8)
  and its KV, decode list 1, prefill context 1. The MTP head needs `embed_tokens` for drafts:
  **(decide)** replicate the embedding on device 1 (2.54 GB, simplest) or gather the needed rows from
  device 0 over P2P per draft step.
- `Control` (pos, pending token): one per device, both advanced each step from the same values; the
  sampled / argmax token produced on device 1 reaches device 0's embedding gather through the
  hand-off path in the other direction (one int per step).

### 3.2 Decode

Per token: device 0 replays list 0 (embed, layers `[0, s)`, hand-off out), device 1 replays list 1
(hand-off in, layers `[s, L)`, head, argmax, token back to device 0). With (a) or (b) the host
submits both lists and waits once on device 1's fence. MTP's draft / verify / commit (spec 8) run on
both devices for the verify (M = K + 1 rows handed off together) and on device 1 for drafts.

### 3.3 Prefill: the chunk pipeline

Two prefill contexts, one per device, and double-buffered hand-off buffers on device 1
(`[2][C][5120]` bf16, ~40 MB). Chunk `c`: device 0 runs `[0, s)` and hands off into buffer
`c mod 2`; device 1 runs `[s, L)` on buffer `c mod 2` while device 0 starts chunk `c + 1`. The last
chunk's last row goes through the head on device 1. Prefill backends (`l0`, `l0-int8`, flash
attention) are unchanged per device.

### 3.4 What else changes

- **Spec 7 snapshots:** each device saves and restores the state of its own layers; the host store's
  entries become `{device 0 part, device 1 part}`; the per-block hook fires on device 1 when the
  block's last layer is done.
- **Spec 9 / 10 / 14 / 15:** per-device kernel variants are the same binaries; `ModelDesc` gains the
  split (`layers_on(device)`).
- **CLI:** `--pp 2` (default 1) and `--pp-split s`; `ZE_AFFINITY_MASK` must expose both cards.
- **The box's GPU lock** (`~/b70-gpu.lock`) is taken once for both cards by a PP job.

## 4. Correctness gates

- **P1, bitwise equal to one card.** PP changes where layers run, not what they compute: the same
  kernels in the same order on the same data, plus a copy. Decode and prefill outputs, logits, KV and
  GDN state must be **bitwise identical** to the single-device run (on device 0) for the golden
  prompts, a 32k prompt, and 64 greedy tokens. Device 1 runs the same binaries; if a kernel's result
  depended on the device it would show here.
- **P2:** the golden gates, replay determinism, `prefill_split_*`, spec 7's C1/C2, spec 8's M2, all
  with `--pp 2`.
- **P3, long context:** Qwen3.8 at max_len 262144 loads with both cards (the memory report per
  device); passkey at 5 / 50 / 95 % of ~250k.
- **P4, failure modes:** a missing second device, P2P not available (`zeDeviceCanAccessPeer` false),
  or a hand-off timeout are refused with a clear error, never a hang.

## 5. Speed bars

Idle box (both cards free of other DRM holders), interleaved pairs against one card, median of 3.

- **P0 (first):** P2P capability and bandwidth both ways (copy engine and kernel peer writes), the
  three hand-off options' latency for 10 KB and 20 MB, and their cost inside a replayed list.
- **S1, decode:** `--pp 2` within **2 %** of one card at 4k and 32k depth (the hand-off must not cost
  more than that).
- **S2, long prefill:** pp32768 and pp65536 **>= 1.7x** one card; pp4096 (two chunks, so little
  overlap) **>= 1.2x** (estimates; P0 refines them).
- **S3, memory:** Qwen3.8 at 262144 fits with both cards (recorded per device).

## 6. Stages

- **16a, probe:** P0 on the box with both cards; the hand-off choice; the split proposal.
- **16b, decode PP:** placement, per-device lists, the hand-off, `--pp 2`; P1 (decode), P2 (decode
  parts), P4; S1.
- **16c, prefill pipeline:** two prefill contexts, double-buffered hand-off, the chunk pipeline;
  P1 (prefill), P2; S2.
- **16d, integration:** spec 7 snapshots per device, MTP across the split, the server flag, P3, S3,
  the record.

## 7. Out of scope

- Tensor parallel (operator, 2026-10-05).
- More than two devices.
- PP combined with batching (two micro-batches in flight): after spec 13.
- Splitting one layer across cards.
