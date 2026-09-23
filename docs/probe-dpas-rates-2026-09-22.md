# DPAS rate by data type on one Arc Pro B70 (2026-09-22)

**Headline.** The B70's XMX array runs **one `dpas.8x8` every 16 EU-cycles
regardless of data type**, so the ops per second scale **exactly** with the
builtin's K depth: bf16 and fp16 (k16) **183.45 TFLOP/s**, int8 and every
4-bit by 8-bit mixed form (k32) **366.9 TIOP/s = 2.000x**, int4, uint4 and int2
(k64) **733.8 TIOP/s = 4.000x**. The bf16 number is **99.97% of the
clock-derived peak**, which is what makes every other row trustworthy.

**FP4 and every microscaling form do not compile on this device at all**, and no
unscaled FP8 entry point is declared.

Every number is labelled **measured** unless marked **derived** or
**estimated**.

## 1. The question

`strings` on the compiler runtime lists `matrix_mad` builtin families for bf16
and fp16 (k16), i8 (k32), i4, i2 and e2m1 (k64), mixed integer forms, e4m3 and
e5m2 (k32), and `scaled_matrix_mad` variants. **A builtin existing in the
compiler is not proof the silicon runs it, still less at the K-implied rate.**
The compiler serves many platforms.

Our production GEMM reaches **161.7 TFLOP/s** on the gate‖up shape at M=2048
(see [probe-fused-dequant-2026-09-22.md](probe-fused-dequant-2026-09-22.md)),
which is the bf16 DPAS rate and is the binding constraint on the whole linear
path. Before designing a W4A8, W4A4 or FP4 kernel, the question is: what rate
does each type actually sustain, and what is the ratio to bf16?

This is a **diagnostic rate probe**, not a benchmark series row.

## 2. Method

A probe-only kernel and host, one binary per type, and one appended build
stanza. **No production file was edited.** No runtime path binds the probe; the
kernel computes a deliberately meaningless dot product.

It is **not a GEMM**: no tiling, no k-loop over memory, no barrier. Every
operand is loaded once into registers before the loop, and the loop body is a
straight chain of 8 x 8 = **64 independent `matrix_mad` calls** on
register-resident operands. Eight independent accumulators round-robin, so
eight `dpas` separate any two writes to the same accumulator, enough to cover
the systolic pipe's latency. Memory traffic over a whole kernel is about 1 KB
of operand loads and one 32-byte store per work-item, against a million or more
`dpas` per sub-group.

**The hazard this is built against** is a loop the compiler deleted, which
would report an impressive and meaningless number. Three things stop it, and
the assembly in section 3 is the proof rather than the intent:

1. operands come from a runtime pointer, so nothing can be constant-folded;
2. every accumulator is summed and **stored unconditionally**, so no `dpas` is
   dead;
3. the trip count is a kernel argument, so the loop cannot be unrolled away or
   evaluated at compile time.

The operand seed `0x3c3c3c3c` reads as a **normal, near-unit value in every
format under test** (bf16 about 0.0115, fp16 about 1.06, int8 60, int4 3 or -4,
int2 0 or -1). No denormals, no NaN, nothing that could make one type's
arithmetic special.

Each binary is built at **256 GRF**, which is mandatory here for the production
GEMM's reason: eight live accumulator vectors plus eight operand registers is
about 112 GRF, which spills at the 128-GRF default and would measure the spill
instead of the pipe. **Zero spill and fill lines in all eight assemblies.**

**Ops per mad**, the figure every rate is derived from:

> ops = 2 x M x N x K, with **M = 8** (the dpas repeat count), **N = 16** (the
> sub-group width) and **K** the builtin's k-suffix. The factor of 2 counts the
> multiply and the add. So k16 gives **4096**, k32 gives **8192**, k64 gives
> **16384** ops per `matrix_mad` per sub-group.

**Protocol.** Per cell: calibrate the trip count to about 40 ms, one discarded
warm-up, then 8 replays of 2 launches each, first 3 discarded, median of the
last 5. Work-group 256, that is 16 sub-groups of 16. Grid swept over 64, 128,
256 and 512 work-groups; the table reports the best cell. Grid 64 is 1024
sub-groups, exactly the device's thread capacity at 256 GRF (**derived**: 256
EUs by 4 threads per EU), so the larger grids are 2x, 4x and 8x
oversubscription. Every type's best cell is at 64 or 128 and reproduces to
0.01%. Three types dip at the oversubscribed grids, reproducibly: see section 6.

**Environment.** One Arc Pro B70, **256 EUs, 32 Xe-cores, max core clock
2800 MHz**, read from Level Zero device properties by the probe itself.
ocloc and compute runtime 26.35.39758.10, IGC 2.41.5.

**Idle observation**, DRM fd holders enumerated immediately before each of
three runs:

| run | DRM fd holders | load avg | card state |
|---|---|---|---|
| 1, sweep | a terminal daemon, plus a test on the other card | 2.63 | neighbour on card 1 |
| 2 | a terminal daemon, plus a test on the other card | 1.27 | neighbour on card 1 |
| **3, sweep, quoted below** | **the terminal daemon only** | **0.80** | **strictly idle** |

The daemon is a desktop session service that holds render-node file descriptors
and submits nothing. The neighbours in runs 1 and 2 were pinned to the other
card, which this probe did not touch. Zero containers throughout.

**Run 3 is the one quoted in section 4**, and it satisfies the record-grade
idle bar for the card under test. Runs 1 and 2 agree with it to four
significant figures anyway. Labelled a **diagnostic** rate regardless.

## 3. Assembly evidence: the loops are DPAS bound

Dumped with shader dumps enabled around the identical `ocloc` command line the
build uses, reading the final Xe2 assembly rather than the intermediate
representation.

For **every one of the eight types**: the single backward-branching loop
contains **exactly 64 `dpas` instructions**, the 64 the source asks for, none
hoisted, none eliminated, none expanded into a sequence, and **zero** `dpas`
anywhere else in the kernel. Every kernel reports 256 GRF and **no spill or
fill**.

| binary | dpas in loop | non-dpas instructions in loop | GRF | spill/fill |
|---|---:|---:|---:|---:|
| `pdr_bf16` | 64 | 5 | 256 | 0 |
| `pdr_f16` | 64 | 5 | 256 | 0 |
| `pdr_i8` | 64 | 4 | 256 | 0 |
| `pdr_i4i8` | 64 | 21 | 256 | 0 |
| `pdr_i8i4` | 64 | 5 | 256 | 0 |
| `pdr_i4` | 64 | 4 | 256 | 0 |
| `pdr_u4` | 64 | 4 | 256 | 0 |
| `pdr_i2` | 64 | 21 | 256 | 0 |

The non-dpas instructions are **only** loop control and scoreboard
synchronisation: `add`, `cmp`, `jmpi`, `sync.nop`, `sync.allwr`. There is no
repacking, no conversion, and no move of operand data inside any loop. The
whole bf16 loop body, verbatim apart from the `dpas` block:

```
_0_009:
          sync.nop  null                             {Compacted,I@1}
          sync.allwr ($3,$4,$5,$6,$7,$8,$9,$14,$15,$16,$17,$18)
          … 64 × dpas.8x8 …
(W)       add  (1|M0)    r1.8<1>:d   r1.8<0;1,0>:d   1:w
(W)       cmp  (16|M0)   (lt)f0.0    null<1>:d  r1.8<0;1,0>:ud  r3.0<0;1,0>:ud  {I@1}
(W&f0.0)  jmpi _0_009
```

**Operand types in the loop, one verbatim line per binary** (operand order is
`dst src0(acc) src1(B) src2(A)`):

```
bf16   dpas.8x8 (16|M0)   r10:f    r10:f    r142:bf   r103.0:bf   {Atomic,Compacted,$2.dst}
f16    dpas.8x8 (16|M0)   r10:f    r10:f    r142:hf   r103.0:hf   {Atomic,Compacted,$2.dst}
i8     dpas.8x8 (16|M0)   r38:d    r38:d    r108:b    r249.0:b    {Atomic,Compacted,$2.dst}
i4i8   dpas.8x8 (16|M0)   r38:d    r38:d    r112:b    r251.0:s4   {$2}
i8i4   dpas.8x8 (16|M0)   r72:d    r72:d    r140:s4   r249.0:b    {Atomic,$2.dst}
i4     dpas.8x8 (16|M0)   r38:d    r38:d    r108:s4   r249.0:s4   {Atomic,Compacted,$2.dst}
u4     dpas.8x8 (16|M0)   r38:d    r38:d    r108:u4   r249.0:u4   {Atomic,Compacted,$2.dst}
i2     dpas.8x8 (16|M0)   r10:d    r10:d    r126:s2   r105.0:s2   {$2}
```

This is the load-bearing part of the evidence. The 4-bit and 2-bit rows carry
**native `:s4`, `:u4` and `:s2` operand encodings on a single `dpas.8x8`**: the
hardware instruction really takes sub-byte operands at the deeper K, the
compiler did not widen them to bytes, and it did not issue two `dpas` for one
builtin. `{Atomic}` on most lines is the back-to-back chaining flag, so the
sequence is being issued as one uninterrupted macro chain.

## 4. Results

Run 3, strictly idle. The ratio column is **derived** from this table's own
bf16 row, not from the production GEMM's number.

| type (builtin family) | K | ops per mad (derived) | best grid | dpas / launch | ms / launch | **ops/s (measured)** | ratio to bf16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bf16_bf16_matrix_mad_k16` | 16 | 4096 | 512 | 1.786e9 | 39.871 | **183.45 TFLOP/s** | **1.000x** |
| `f16_f16_matrix_mad_k16` | 16 | 4096 | 512 | 1.785e9 | 39.859 | **183.45 TFLOP/s** | 1.000x |
| `i8_i8_matrix_mad_k32` | 32 | 8192 | 64 | 1.735e9 | 38.731 | **366.90 TIOP/s** | **2.000x** |
| `i4_i8_matrix_mad_k32` | 32 | 8192 | 128 | 1.766e9 | 39.433 | **366.89 TIOP/s** | **2.000x** |
| `i8_i4_matrix_mad_k32` | 32 | 8192 | 64 | 1.733e9 | 38.700 | **366.90 TIOP/s** | 2.000x |
| `i4_i4_matrix_mad_k64` | 64 | 16384 | 64 | 1.729e9 | 38.605 | **733.80 TIOP/s** | **4.000x** |
| `u4_u4_matrix_mad_k64` | 64 | 16384 | 64 | 1.733e9 | 38.697 | **733.80 TIOP/s** | 4.000x |
| `i2_i2_matrix_mad_k64` | 64 | 16384 | 512 | 1.754e9 | 39.157 | **733.79 TIOP/s** | 4.000x |

**One model explains the whole table** (derived): divide each row's rate by its
ops per mad and every type lands on the **same issue rate, 4.479e10 `dpas.8x8`
per second**, within 0.01%. With 256 EUs at 2.8 GHz, that is 7.168e11 EU-cycles
per second, so **16.0 EU-cycles per `dpas.8x8`**, or 128 MACs per EU per cycle,
**2048 bf16 FLOP per Xe-core per cycle**.

The array's depth scaling from K=16 to K=64 is **free**; its width is not.
Nothing in this table is a rate difference between types. It is one constant
issue rate seen through three different K.

### Sanity anchors

- **Clock-derived bf16 peak** (derived): 32 Xe-cores x 2048 FLOP/clk x 2.8 GHz
  = **183.50 TFLOP/s**. Measured **183.45**, which is **99.97%** of it. The
  loop is DPAS bound with essentially nothing left on the table, so the ratios
  above are ratios of the pipe and not of some other bottleneck.
- **Production GEMM** at gate‖up M=2048 measures **161.7 TFLOP/s**, which is
  **88.1%** of this micro-benchmark (derived). The micro-benchmark sits
  **above** the production kernel, as it must; the 11.9% gap is the GEMM's
  memory, barrier and epilogue cost.

## 5. What is not available on this compiler and device

All four were attempted. Output verbatim.

**FP4 (`e2m1_e2m1_matrix_mad_k64`, scaled and unscaled): refused by the
backend.** Not a missing declaration. It type-checks, then the backend rejects
it by name.

```
[bmg-g31] error: FP4 Dpas instruction is not supported on this device!
[bmg-g31] error: backend compiler failed build.
```

**Every `scaled_matrix_mad` form: internal compiler error.** Tried
`bf16_bf16_scaled_matrix_mad_k16`, `f16_f16_scaled_matrix_mad_k16`,
`hf8_hf8_scaled_matrix_mad_k32`, `bf8_bf8_scaled_matrix_mad_k32` and
`e2m1_e2m1_scaled_matrix_mad_k64`. All five crash the compiler, and the crash
**reproduces from a kernel containing a single call**, so it is the builtin and
not this probe's 64-dpas loop.

```
internal compiler error, abnormal program termination
[bmg-g31] IGC: Internal Compiler Error: Abnormal termination
Build failed with error code: -11
```

**FP8 unscaled: no OpenCL C entry point exists.**
`intel_sub_group_e4m3_e4m3_matrix_mad_k32`, the `e5m2` pair, the `hf8` pair and
the `bf8` pair are all

```
error: use of undeclared identifier 'intel_sub_group_e4m3_e4m3_matrix_mad_k32'
```

The compiler's own header declares **only** the *scaled* FP8 forms, and those
are exactly the ones that crash the backend. So there is **no reachable FP8
matrix-multiply path in OpenCL C on this device at all**, by either spelling.

This is the concrete answer to the framing question. `strings` lists
`e4m3_e4m3_matrix_mad_k32` and `e2m1_e2m1_matrix_mad_k64`; **neither is a
capability of this silicon.** A name in `strings` proves only that the compiler
knows the name.

**Not a gap, but worth recording:** `i2_i2_matrix_mad_k64` measures **733.79
TIOP/s, the same as `i4_i4`'s 733.80**. The k-suffix stops at 64, so 2-bit
operands raise nothing over 4-bit. K=64 is this array's floor on element width.

## 6. Reproducibility, and one honest anomaly

Three full runs; runs 1 and 3 printed the whole grid sweep. Best cell per type:

| type | run 1 | run 2 | run 3, idle | spread |
|---|---:|---:|---:|---:|
| `bf16_bf16_k16` | 183.44 | 183.44 | 183.45 | 0.01% |
| `f16_f16_k16` | 183.45 | 183.44 | 183.45 | 0.01% |
| `i8_i8_k32` | 366.87 | 366.86 | 366.90 | 0.01% |
| `i4_i8_k32` | 366.87 | 366.90 | 366.89 | 0.01% |
| `i8_i4_k32` | 366.88 | 366.90 | 366.90 | 0.01% |
| `i4_i4_k64` | 733.74 | 733.80 | 733.80 | 0.01% |
| `u4_u4_k64` | 733.81 | 733.81 | 733.80 | 0.00% |
| `i2_i2_k64` | 733.81 | 733.81 | 733.79 | 0.00% |

The ceiling is solid: every type reproduces to **0.01%** across a busy-neighbour
run and a strictly idle one.

**The anomaly, recorded rather than smoothed over.** Within each sweep, 24 of
the 32 grid cells sit within 0.02% of their type's best, but **three types,
`i8_i8`, `i4_i4` and `u4_u4`, dip at the oversubscribed grids, and the dip
reproduces on the strictly idle run**:

| type, run 3 | grid 64 | grid 128 | grid 256 | grid 512 |
|---|---:|---:|---:|---:|
| `i8_i8_k32` | 366.90 | 366.90 | **334.53 (-8.8%)** | **355.21 (-3.2%)** |
| `i4_i4_k64` | 733.80 | 733.80 | **688.52 (-6.2%)** | **710.40 (-3.2%)** |
| `u4_u4_k64` | 733.80 | 733.80 | **669.88 (-8.7%)** | **710.42 (-3.2%)** |
| the other five types | | | no dip | no dip |

So the first reading, transient interference from the neighbour on the other
card, was **wrong**: it reproduces with the box idle. The grid-512 dip is a
stable -3.19% for all three, about 1/32 of the device; the grid-256 dip moves
between -6.2% and -8.8% run to run. It does not track the mixed or unmixed
distinction, the K depth, or the accumulator type, and the three affected
binaries are exactly the three whose loops carry the fewest scoreboard `sync`
instructions, 4 against 5 or 21.

**The cause was not chased.** It costs nothing here, because it only ever
lowers a cell below the ceiling the same type reaches at grid 64 and 128, and
the ceiling is what this probe is for. Anyone writing a real int8 or int4 kernel
should know the effect exists and sweep occupancy rather than assume.

## 7. What this implies for W4A8, W4A4 and FP4, rate only

Strictly the DPAS rate. **Nothing here is an accuracy claim, a memory-traffic
claim, or a recommendation to adopt anything.**

**W4A8: no rate advantage over W8A8 whatsoever.** `i4_i8_k32` measures
**366.89 TIOP/s** and `i8_i4_k32` **366.90**, against `i8_i8_k32`'s **366.90**,
the same number to within 0.01%, because the mixed builtin's K is **32, the
same as int8's**. The 4-bit operand shortens no part of the pipe. The DPAS
ceiling of a W4A8 linear path is therefore **2.000x bf16**, identical to a W8A8
one. Whatever W4A8 is worth on this device, **none of it comes from the math
pipe**; the case would have to be made entirely on weight bytes, and that case
is not made here.

Note also which operand is 4-bit: in a GEMM the weights are the B operand, so
the builtin a W4A8 kernel would actually issue is `i8_i4_k32`, and it measures
the same. That prediction was tested directly and the kernel it describes was
built and rejected on other grounds: see
[probe-w4a8-2026-09-23.md](probe-w4a8-2026-09-23.md).

**W4A4: the only shape that moves the ceiling past 2x.** **733.80 TIOP/s =
4.000x bf16**, and it is a genuine single-instruction rate: section 3 shows
`:s4` on both operands of one `dpas.8x8`. If a W4A4 kernel reached the
production GEMM's 88.1% efficiency, the **estimated** in-kernel rate would be
about 646 TIOP/s, an estimate by proportion with no W4A4 kernel written and no
accuracy question addressed. The unsigned form measures identically, so the
packing a real asymmetric zero-point checkpoint uses costs nothing. Going below
4 bits buys nothing further.

**FP4: not available, so there is no rate to design against.** The backend
refuses every `e2m1` form by name, and every microscaling entry point crashes
the compiler. An FP4 path on this device would have to be emulated, decoding
`e2m1` to int4 or bf16 in the ALU and issuing `i4_i4_k64` or the bf16 form, in
which case its rate is by construction the int4 or bf16 rate already in the
table above, minus the ALU cost of the decode, and the block scales would have
to be applied outside the `dpas`. **For rate purposes, FP4 on this B70 is int4
with extra work.**

**The whole rate landscape.** The linear path's current constraint is bf16 at
161.7 TFLOP/s in situ against a 183.45 TFLOP/s pipe ceiling. The only DPAS
lever on this device worth more than 2x is **4-bit by 4-bit**; everything
between, meaning int8 and W4A8 in either operand order, is one 2x step, and
everything below 4 bits is flat. The design question is then entirely about
whether a 4x4 shape is reachable at acceptable accuracy, which this probe says
nothing about and the W4A8 probe says a good deal about.

## 8. Files

- `tools/probe/probe_dpas_rate.cl`: probe-only kernel, one binary per type.
- `tools/probe/probe_dpas_rate.cc`: plain g++ and Level Zero, a queue, a fence
  and a closed regular command list. No SYCL, no prefill-runtime dependency.
- `tools/probe/CMakeLists.txt`: one appended stanza. The types the backend
  refuses are deliberately absent, with the reason recorded in the stanza.
- No production file was edited.

```
ZE_AFFINITY_MASK=0 tools/box.sh run './build/tools/probe/probe_dpas_rate --sweep'
```
