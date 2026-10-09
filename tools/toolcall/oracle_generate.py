#!/usr/bin/env python3
"""Greedy bf16 baseline for the tool-call set (spec 5 T0, gate A4), on CPU.

    oracle_generate.py [--model kolibri|qwen4exp] [--device cpu|cuda|xpu] <snapshot> <set dir> <out dir>
                       [--new-tokens N] [--batch 1|N|auto] [--resident none|auto] [--only a,b,..]

Builds transformers' Qwen3_5ForCausalLM the way tools/rotate/check_rotation.py
does (text_config, load_sd in bf16, eager attention), then for every scenario
in <set dir>/manifest.json, after checking its .ids file's SHA-256 against the
manifest, runs greedy generate with 192 new tokens and writes
<out>/<name>.bf16.ids (the new ids) and <out>/<name>.bf16.txt (decoded with
the snapshot's tokenizer, special tokens kept).

Resumable: a scenario whose two outputs exist is skipped, because the whole set
takes hours. Outputs are written to a temporary name and renamed, so a killed
run never leaves a half-written file behind.

Run inside the reference container (tools/oracle/run_in_container.sh with
ORACLE_MODEL=models--Qwen--Qwen3.8-27B).

The model follows config.json's model_type:
  qwen3_5 / agnes   transformers' Qwen3_5ForCausalLM on the bf16 checkpoint (above; spec 14
                    for Agnes), HF generate;
  qwen3_5_moe       Ornith 1.5 (spec 15e Task 3): tools/oracle/ornith_ref.py's layer-streamed
                    Qwen3_5MoeForCausalLM on its INT4 checkpoint dequantised by the repo's one
                    rule (15a: the engine's own weights, so a mismatch is the engine's, not the
                    quantisation's), greedy through its KV cache;
  k2_horizon        K2-Horizon (spec 18d, K4): tools/oracle/k2_ref.py's port of
                    modeling_k2_horizon.py, layer at a time, mode bf16 (bitwise the vendored
                    model in bf16, eager attention), on the checkpoint given - the int4 one
                    (dequantised, 18a's reference) or the bf16 original - greedy through its
                    own cache;
  kolibri1          Kolibri-1 (spec 20e, KL4; `--model kolibri` says so and checks it):
                    tools/oracle/kolibri_ref.py's KolibriRef, layer at a time, mode bf16, on the
                    checkpoint given - the 156 GB bf16 source (the KL4 reference, wherever 20b
                    runs: `--device` cuda / xpu where there is one) or the int4 export
                    (dequantised) - greedy through its own cache, the head's 128000 rows.
  qwen4_exp         Qwen3.8-Flash-Next (spec 21e; `--model qwen4exp` says so and checks it):
                    tools/oracle/qwen4exp_ref.py's layer-streamed model (transformers 5.19.0: the
                    qwen4exp site first on PYTHONPATH) on the checkpoint given - Intel's int4 g128
                    experts dequantised, its bf16 dense layers, the PLE rows from <snapshot>-ple-int8
                    when present (the engine's format; Q4_PLE overrides) - greedy through its own
                    cache; Q4_LAYERS=N truncates as the engine's --layers N does.
The MoE paths stop after an EOS id of generation_config.json (kept, as HF generate keeps it)
or --new-tokens ids (default 192; K2's A4 run uses 512: its replies open with reasoning), and
also write <out>/<name>.bf16.gap: each greedy step's top-1 minus top-2 logit (compare_ref.py).

The MoE paths' batched mode (--batch N, or auto from the memory plan; default 1 = one scenario
at a time, the code path as before): up to N scenarios in flight, LAYER-MAJOR - every sequence
through layer 0, then layer 1, ... so each layer's weights are dequantised once per pass for all
of them - while each sequence runs alone on its own [1, T] tensors and cache (no op sees two
sequences: no padding, no batch-shaped GEMM), so its ids and logits are bitwise the sequential
run's (test_oracle_batched.py). A finished scenario is written at once (resumable as before) and
its slot refilled. --resident auto keeps dequantised weights resident once read (every layer's
dense part, then whole layers of experts) in what the plan leaves; the values are the same.
The plan (ESTIMATES: per-sequence KV / state, the largest forward, the base, against the
container's cgroup cap) is printed first. --only runs just the named scenarios (a cross-check).
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "bf16"
NEW_TOKENS = 192
Q4_TYPES = ("qwen4_exp", "qwen4_exp_text")   # spec 21e: Qwen3.8-Flash-Next (a checkpoint, a text-only export)


def load_manifest(set_dir: str) -> list[dict]:
    with open(os.path.join(set_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def checked_ids(set_dir: str, entry: dict) -> list[int]:
    with open(os.path.join(set_dir, f"{entry['name']}.ids"), "rb") as f:
        raw = f.read()
    got = hashlib.sha256(raw).hexdigest()
    if got != entry["sha256"]:
        sys.exit(f"FATAL: {entry['name']}.ids SHA-256 {got} != manifest {entry['sha256']}")
    ids = [int(x) for x in raw.split()]
    if len(ids) != entry["ids"]:
        sys.exit(f"FATAL: {entry['name']}.ids has {len(ids)} ids, manifest says {entry['ids']}")
    return ids


def write_atomic(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def greedy(step, prompt: list[int], n: int, eos: set[int], argmax) -> list[int]:
    """The MoE paths' greedy loop: `step(ids, pos)` runs ids at positions pos.. through the
    model's cache and returns the last row's logits; `argmax(row)` -> int. Returns the new ids:
    at most n, ending at the first EOS id (kept)."""
    out: list[int] = []
    row = step(prompt, 0)
    while len(out) < n:
        nxt = int(argmax(row))
        out.append(nxt)
        if nxt in eos or len(out) == n:
            break
        row = step([nxt], len(prompt) + len(out) - 1)
    return out


def greedy_batched(step_many, new_state, jobs, batch: int, eos: set[int], argmax, on_done,
                   on_row=None, log=None, on_start=None) -> None:
    """greedy() over several sequences at once (--batch): jobs = [(name, prompt ids, n)], at most
    `batch` in flight. Each pass hands step_many([(ids, pos, state)]) one item per sequence in
    flight - exactly what greedy() would feed that sequence next: its prompt at 0 on its first
    pass, then its last new id at the next position - and gets each item's last logits row back.
    A sequence that ends (an EOS id, kept, or its n ids) goes to on_done(name, new ids) at once and
    its slot is refilled from the queue, so a new prompt's forward shares a layer pass with the
    others' decode steps. Per sequence the ids are greedy()'s whenever step_many's row for an item
    is what the sequential step returns for it (the references' forward_many / layer_major
    guarantee that bitwise). on_row(name, row) sees every row (the tests)."""
    queue = list(jobs)
    live: list[dict] = []
    passes = 0
    while queue or live:
        while queue and len(live) < batch:
            name, ids, n = queue.pop(0)
            if on_start:
                on_start(name)
            if n <= 0:
                on_done(name, [])
                continue
            live.append({"name": name, "prompt": list(ids), "n": n, "out": [], "state": new_state(),
                         "feed": (list(ids), 0)})
        if not live:
            break
        t = time.time()
        rows = step_many([(a["feed"][0], a["feed"][1], a["state"]) for a in live])
        passes += 1
        n_pr = sum(1 for a in live if not a["out"])
        if log and (n_pr or passes % 25 == 0):
            log(f"  pass {passes}: {len(live)} sequences ({n_pr} prompt{'s' if n_pr != 1 else ''}, "
                f"{len(live) - n_pr} decode steps), {time.time() - t:.1f}s")
        keep = []
        for a, row in zip(live, rows):
            if on_row:
                on_row(a["name"], row)
            nxt = int(argmax(row))
            a["out"].append(nxt)
            if nxt in eos or len(a["out"]) == a["n"]:
                a["state"] = None                      # its cache goes now, not at the end
                on_done(a["name"], a["out"])
                continue
            a["feed"] = ([nxt], len(a["prompt"]) + len(a["out"]) - 1)
            keep.append(a)
        live = keep


GiB = 2 ** 30


def mem_model(mtype: str, cfg: dict) -> dict:
    """The batched run's memory from config.json, in bytes - ESTIMATES, the dominant terms only:
      base           what stays whatever the batch: embed and head, the streamed layer and its
                     prefetch, Ornith's shared expert buffers / K2's one layer of experts;
      transient(T)   the largest single-sequence forward's working set (a T-id prompt: the head's
                     [T, V] logits, eager attention's [heads, T, T] scores); sequences run one at
                     a time inside a pass, so one is live at a time;
      per_seq(T)     a sequence of T ids in flight: its KV cache / recurrent state, the hidden rows
                     a pass carries;
      dense          [bytes per layer] of the dequantised non-expert weights (what --resident keeps);
      experts        [bytes per layer] of every routed expert dequantised (0: no experts there)."""
    if mtype == "qwen3_5_moe":
        c = cfg.get("text_config", cfg)
        L, H, V = c["num_hidden_layers"], c["hidden_size"], c["vocab_size"]
        types = c.get("layer_types") or ["full_attention" if (i + 1) % c.get("full_attention_interval", 4) == 0
                                         else "linear_attention" for i in range(L)]
        nf = types.count("full_attention")
        nl = L - nf
        hk, dk, hv, dv = (c["linear_num_key_heads"], c["linear_key_head_dim"], c["linear_num_value_heads"],
                          c["linear_value_head_dim"])
        conv = 2 * hk * dk + hv * dv
        E, I, Is = c["num_experts"], c["moe_intermediate_size"], c.get("shared_expert_intermediate_size", 0)
        heads, kvh, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        common = 3 * H * Is + H + E * H
        gdn = H * conv + H * hv * dv + 2 * H * hv + hv * dv * H
        fa = H * 2 * heads * hd + 2 * H * kvh * hd + heads * hd * H
        dense = [2 * (common + (fa if t == "full_attention" else gdn)) for t in types]
        ex = E * 3 * I * H * 2
        state = nl * (hv * dk * dv * 4 + conv * c.get("linear_conv_kernel_dim", 4) * 2)
        return {"base": 2 * V * H * 2 + ex + 2 * max(dense),
                "transient": lambda T: T * V * 2 + heads * T * T * 8 + 12 * hv * T * max(dk, dv) * 4,
                "per_seq": lambda T: state + T * nf * 2 * kvh * hd * 2 + T * H * 2,
                "dense": dense, "experts": [ex] * L,
                "what": f"KV bf16 in {nf} full-attention layers + GDN state fp32 in {nl} ({state / 2**20:.0f} MiB)"}
    if mtype in Q4_TYPES:               # spec 21e: Qwen3.8-Flash-Next (qwen4exp_ref.py, bf16 dense, eager attention)
        c = cfg.get("text_config", cfg)
        L, H, V = c["num_hidden_layers"], c["hidden_size"], c["vocab_size"]
        hc = c.get("hc_count", 4)
        types = c.get("layer_types") or ["linear_attention" if i % 4 != 3 else "indexed_attention" for i in range(L)]
        nq = sum(1 for t in types[:L] if t != "linear_attention")
        ng = L - nq
        heads, kvh, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        hv, dv, dk = c["linear_num_value_heads"], c["linear_value_head_dim"], c["linear_key_head_dim"]
        E, I, Is = c["num_experts"], c["moe_intermediate_size"], c.get("shared_expert_intermediate_size", 0)
        low = c.get("hc_lowrank", 320)
        hcl = 2 * 2 * (hc * H * (low + hc) + low * hc * H)       # both gated residuals of a layer, bf16
        qsa = H * (2 * heads * hd + 2 * kvh * hd) + heads * hd * H + H * 5 * c.get("index_head_dim", 128)
        gdn = H * (2 * c["linear_num_key_heads"] * dk + 2 * hv * dv) + 2 * H * hv + hv * dv * H
        common = H * (E + 1) + 3 * H * Is
        dense = [hcl + 2 * (common + (gdn if t == "linear_attention" else qsa)) for t in types[:L]]
        ex = E * 3 * I * H * 2
        state = ng * (hv * dk * dv * 4 + (2 * c["linear_num_key_heads"] * dk + hv * dv) * 4 * 2)
        return {"base": 2 * V * H * 2 + ex + 2 * max(dense),
                "transient": lambda T: T * V * 2 + heads * T * T * 8 + T * hc * H * 2 * 8,
                "per_seq": lambda T: state + T * nq * (2 * kvh * hd * 2 + c.get("index_head_dim", 128) * 2) + T * hc * H * 2,
                "dense": dense, "experts": [ex] * L,
                "what": f"KV bf16 + indexer keys in {nq} QSA layers + GDN state fp32 in {ng} ({state / 2**20:.0f} MiB)"}
    L, H, V = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["vocab_size"]
    heads, kvh = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim") or H // heads
    E, I = cfg["num_experts"], cfg["moe_intermediate_size"]
    if mtype == "k2_horizon":           # k2_ref keeps K / V in fp32 tensors (bf16 values)
        mlp_only = set(cfg.get("mlp_only_layers") or [])
        step = cfg.get("decoder_sparse_step", 1) or 1
        sparse = [i not in mlp_only and E > 0 and (i + 1) % step == 0 for i in range(L)]
        mE = cfg.get("mova_num_experts", 0)
        attn = 2 * heads * hd * H + kvh * hd * H + (heads * hd * H if cfg.get("attention_gate_func") else 0)
        dense = [2 * (attn + (mE * H if mE else kvh * hd * H) + E * H + 3 * H * I * cfg.get("num_shared_experts", 0))
                 if s else 2 * (attn + kvh * hd * H + 3 * H * cfg.get("intermediate_size", 0)) for s in sparse]
        ex = 2 * (E * 3 * I * H + mE * kvh * hd * H)
        return {"base": 2 * V * H * 2 + 2 * max(dense) + ex,
                "transient": lambda T: T * V * 6 + heads * T * T * 4 * 4,
                "per_seq": lambda T: T * L * (2 * kvh * hd * 4 + 8) + T * H * 4,
                "dense": dense, "experts": [ex if s else 0 for s in sparse],
                "what": f"KV fp32 in {L} layers"}
    types = cfg.get("layer_types") or []
    ns = types.count("sliding_attention")
    W = (cfg.get("sliding_window") or 0) if cfg.get("use_sliding_window", True) else 0
    nf = L - ns
    layer = 2 * (2 * H * heads * hd + 2 * H * kvh * hd + 3 * H * cfg.get("shared_expert_intermediate_size", I))
    return {"base": 2 * V * H * 2 + E * 3 * I * H * 2 + 2 * layer,
            "transient": lambda T: V * H * 4 + T * V * 4 + heads * T * T * 8,
            "per_seq": lambda T: (T * nf + ns * min(T, max(W - 1, 0))) * 2 * kvh * hd * 2 + T * H * 2,
            "dense": [], "experts": [],          # kolibri_ref: no --resident (its source is 156 GB bf16)
            "what": f"KV bf16 in {nf} full + {ns} sliding layers (window {W})"}


def mem_limit() -> int | None:
    """ORACLE_MEM_GB if set, else the container's memory cap (cgroup v2 / v1), else MemAvailable
    now (an uncapped container on a shared host: the box); None when unknown."""
    if os.environ.get("ORACLE_MEM_GB"):
        return int(float(os.environ["ORACLE_MEM_GB"]) * GiB)
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(p, encoding="utf-8") as f:
                v = f.read().strip()
        except OSError:
            continue
        if v.isdigit() and int(v) < 2 ** 60:
            return int(v)
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


# auto's most sequences in flight. Ornith's time is mostly per-sequence compute (a prompt forward
# is minutes, and a pass of B prompts holds every other sequence for all B), so a few; K2's is
# mostly the per-step dequant a pass shares, so more.
MAX_AUTO_BATCH = {"qwen3_5_moe": 4, "k2_horizon": 8, "kolibri1": 4, "qwen4_exp": 4, "qwen4_exp_text": 4}


def batch_plan(mtype: str, cfg: dict, lens: list[int], n: int, cap: int | None, want="auto",
               keep="none") -> dict:
    """The memory plan: {"batch", "keep_dense", "keep_experts" (layer counts), "text"}.

    want: "auto" or a count. auto = the most sequences in flight whose KV / state at their longest
    (the longest prompt + n ids) fits beside the base and the largest single forward in 85% of the
    cap less 2 GiB (page cache, allocator slack), at most MAX_AUTO_BATCH[mtype] and the scenarios
    left. keep "auto": what then remains goes to residency - first the dense weights of every
    layer (cheap), then whole layers of dequantised experts, layer 0 up; "none": nothing kept."""
    mm = mem_model(mtype, cfg)
    Tm = max(lens) + n
    per, base, tr = mm["per_seq"](Tm), mm["base"], mm["transient"](max(lens))
    room = (0.85 * cap - 2 * GiB - base - tr) if cap else None
    fit = (max(1, int(room // per)) if room > 0 else 1) if room is not None else None
    if want == "auto":
        b = min(MAX_AUTO_BATCH.get(mtype, 4), len(lens), fit if fit is not None else 2)
    else:
        b = max(1, min(int(want), len(lens)))
    kd = ke = 0
    kept = 0
    left = room - b * per if room is not None else None
    if keep == "auto" and left is not None and left > 0:
        for d in mm["dense"]:
            if kept + d > left:
                break
            kept += d
            kd += 1
        if kd == len(mm["dense"]):
            for e in mm["experts"]:
                if not e:
                    continue
                if kept + e > left:
                    break
                kept += e
                ke += 1
    text = ((f"memory plan (ESTIMATES): cap {cap / GiB:.1f} GiB" if cap else "memory plan (ESTIMATES): cap unknown")
            + f"; base {base / GiB:.1f} GiB (embed + head, streamed layer + prefetch, expert buffers), largest "
            f"forward {tr / GiB:.1f} GiB ({max(lens)}-id prompt), per sequence {per / GiB:.2f} GiB at {Tm} ids "
            f"({mm['what']}); fits {fit if fit is not None else '?'} -> batch {b}"
            f"{' (auto, at most %d)' % MAX_AUTO_BATCH.get(mtype, 4) if want == 'auto' else ''}")
    if mm["dense"]:
        n_ex = sum(1 for x in mm["experts"] if x)
        text += (f"; resident {keep}: dense weights of {kd}/{len(mm['dense'])} layers, experts of {ke}/{n_ex} layers "
                 f"({kept / GiB:.1f} GiB)")
    text += f"; peak ~{(base + tr + b * per + kept) / GiB:.1f} GiB"
    if fit is not None and want != "auto" and b > fit:
        text += f"  WARNING: {b} > the {fit} that fit the cap"
    return {"batch": b, "keep_dense": kd, "keep_experts": ke, "text": text}


def moe_runner(snap: str, mtype: str, device: str = "cpu", keep: tuple[int, int] = (0, 0)):
    """-> a namespace for Ornith (qwen3_5_moe), K2-Horizon (k2_horizon) or Kolibri-1 (kolibri1;
    `device` is its reference's): generate(ids, n, eos[, on_row]) - the sequential path, one
    scenario at a time - and new_state() / step_many(items) - the layer-major batched path
    (greedy_batched); `ref` is the reference object (K2Ref / KolibriRef / the streamed model).
    keep = (dense layers, expert layers) resident once dequantised (batch_plan's counts: the
    first that many layers; expert layers among those with experts); Ornith and K2 only."""
    import torch
    from types import SimpleNamespace

    def load(name: str):
        spec = importlib.util.spec_from_file_location(f"b70_{name}", os.path.join(HERE, "..", "oracle", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    def argmax(row):
        return int(torch.argmax(row).item())

    def watch(am, on_row):        # generate(..., on_row): every logits row the greedy loop reads (tests)
        return am if on_row is None else (lambda row: (on_row(row), am(row))[1])

    if mtype == "kolibri1":
        KR = load("kolibri_ref")
        src = KR.Checkpoint(snap)
        ref = KR.KolibriRef(src.cfg, src, mode="bf16", device=device)
        print(f"Kolibri-1 reference: kolibri_ref.py, {src.cfg.num_hidden_layers} layers, mode bf16, "
              f"device {device}", flush=True)

        def generate(ids, n, eos, on_row=None):
            cache = ref.new_cache()
            return greedy(lambda chunk, pos: ref.forward(chunk, pos, cache)[-1], ids, n, eos, watch(argmax, on_row))
        return SimpleNamespace(generate=generate, new_state=ref.new_cache, argmax=argmax,
                               step_many=ref.forward_many, ref=ref)

    if mtype in Q4_TYPES:
        # Spec 21e: Qwen3.8-Flash-Next - tools/oracle/qwen4exp_ref.py's layer-streamed model (transformers 5.19.0 from
        # the qwen4exp site first on PYTHONPATH), the checkpoint's experts as stored (Intel's int4 g128 dequantised), the
        # PLE rows from the engine's int8 file when it sits beside the snapshot (21b's <snapshot>-ple-int8: the engine-
        # format reference, spec 21 F3) - Q4_PLE overrides (bf16 | int8:<dir>) - greedy through its own cache. The
        # batched path: layer_major's pre-mixer rows -> final_logits of each sequence's LAST row (the sequential step
        # asks for the last row's logits only: logits_to_keep=1, the same one-row head).
        Q4 = load("qwen4exp_ref")
        tc = Q4.text_config(snap, int(os.environ["Q4_LAYERS"]) if os.environ.get("Q4_LAYERS") else None)
        side = snap.rstrip("/") + "-ple-int8"
        ple_spec = os.environ.get("Q4_PLE") or (f"int8:{side}" if os.path.isdir(side) else "bf16")
        ple = Q4.PleTable(Q4.ple_source(snap, ple_spec), tc)
        kd = range(min(keep[0], tc.num_hidden_layers))
        ke = range(min(keep[1], tc.num_hidden_layers))
        model, pf, lazy = Q4.build_streamed(snap, tc, ple, keep_dense=kd, keep_experts=ke)
        print(f"Qwen3.8-Flash-Next reference: qwen4exp_ref.py, {tc.num_hidden_layers} layers, streamed, PLE {ple.kind}; "
              f"resident once read: dense {len(kd)} layers, experts {len(ke)} layers", flush=True)
        major = Q4.layer_major(model, pf)

        def generate(ids, n, eos, on_row=None):
            state = {"cache": None}

            def step(chunk, pos):
                with torch.no_grad():
                    o = model(input_ids=torch.tensor([chunk]), past_key_values=state["cache"], use_cache=True,
                              logits_to_keep=1)
                state["cache"] = o.past_key_values
                return o.logits[0, -1].float()
            return greedy(step, ids, n, eos, watch(argmax, on_row))

        def step_many(items):
            return [Q4.final_logits(model, R[-1:])[0].float() for R in major(items)]
        return SimpleNamespace(generate=generate, new_state=lambda: {"cache": None}, argmax=argmax,
                               step_many=step_many, lazy=lazy, ref=model, pf=pf)

    if mtype == "k2_horizon":
        K2 = load("k2_ref")
        src = K2.Checkpoint(snap)
        c = src.cfg
        kd = range(min(keep[0], c.num_hidden_layers))
        ke = [i for i in range(c.num_hidden_layers) if c.is_sparse(i)][:keep[1]]
        ref = K2.K2Ref(c, src, mode="bf16", keep_dense=kd, keep_experts=ke)
        print(f"K2-Horizon reference: k2_ref.py, {c.num_hidden_layers} layers, mode bf16, "
              f"{'int4 g%d dequantised' % c.group_size if c.group_size else 'bf16'}; resident once read: "
              f"dense {len(kd)} layers, experts {len(ke)} layers", flush=True)

        def generate(ids, n, eos, on_row=None):
            cache = ref.new_cache()
            return greedy(lambda chunk, pos: ref.forward(chunk, pos, cache)[-1], ids, n, eos, watch(argmax, on_row))
        return SimpleNamespace(generate=generate, new_state=ref.new_cache, argmax=argmax,
                               step_many=ref.forward_many, ref=ref)

    OR = load("ornith_ref")
    tc = OR.text_config(snap)
    kd = range(min(keep[0], tc.num_hidden_layers))
    ke = range(min(keep[1], tc.num_hidden_layers))
    model, pf, lazy = OR.build_streamed(snap, tc, keep_dense=kd, keep_experts=ke)
    print(f"Ornith reference: ornith_ref.py, {tc.num_hidden_layers} layers, int4 dequantised, streamed; "
          f"resident once read: dense {len(kd)} layers, experts {len(ke)} layers", flush=True)

    def generate(ids, n, eos, on_row=None):
        state = {"cache": None}

        def step(chunk, pos):
            with torch.no_grad():
                o = model(input_ids=torch.tensor([chunk]), past_key_values=state["cache"], use_cache=True)
            state["cache"] = o.past_key_values
            return o.logits[0, -1].float()
        return greedy(step, ids, n, eos, watch(argmax, on_row))
    return SimpleNamespace(generate=generate, new_state=lambda: {"cache": None}, argmax=argmax,
                           step_many=OR.layer_major(model, pf), lazy=lazy, ref=model, pf=pf)


MOE_TYPES = ("qwen3_5_moe", "k2_horizon", "kolibri1", "qwen4_exp", "qwen4_exp_text")
# --model NAME -> the model_type(s) it must be (spec 21e: qwen4exp - the checkpoints say qwen4_exp, a text-only
# export qwen4_exp_text)
MODELS = {"kolibri": ("kolibri1",), "qwen4exp": ("qwen4_exp", "qwen4_exp_text")}


def parse_args(argv: list[str]) -> dict:
    """[--model kolibri] [--device D] <snapshot> <set dir> <out dir> [--new-tokens N]
    [--batch N|auto] [--resident none|auto] [--only NAME,...] -> a dict."""
    a = {"model": None, "device": "cpu", "new_tokens": NEW_TOKENS, "batch": "1", "keep": "none", "only": None}
    pos: list[str] = []
    i = 0
    while i < len(argv):
        x = argv[i]
        if x in ("--model", "--device", "--new-tokens", "--batch", "--resident", "--only"):
            if i + 1 >= len(argv):
                sys.exit(f"{x} needs a value")
            v = argv[i + 1]
            if x == "--model":
                if v not in MODELS:
                    sys.exit(f"--model expects one of {', '.join(MODELS)}, got {v!r}")
                a["model"] = v
            elif x == "--device":
                a["device"] = v
            elif x == "--batch":
                if v != "auto" and not (v.isdigit() and int(v) >= 1):
                    sys.exit(f"--batch expects a count >= 1 or auto, got {v!r}")
                a["batch"] = v
            elif x == "--resident":
                if v not in ("none", "auto"):
                    sys.exit(f"--resident expects none or auto, got {v!r}")
                a["keep"] = v
            elif x == "--only":
                a["only"] = [n for n in v.split(",") if n]
            else:
                a["new_tokens"] = int(v)
            i += 2
            continue
        pos.append(x)
        i += 1
    if len(pos) != 3:
        sys.exit(__doc__)
    a["snapshot"], a["set_dir"], a["out"] = pos
    return a


def main() -> None:
    a = parse_args(sys.argv[1:])
    snap, set_dir, out, new_tokens = a["snapshot"], a["set_dir"], a["out"], a["new_tokens"]
    os.makedirs(out, exist_ok=True)
    manifest = load_manifest(set_dir)
    if a["only"]:
        unknown = sorted(set(a["only"]) - {e["name"] for e in manifest})
        if unknown:
            sys.exit(f"--only: {unknown} not in {set_dir}/manifest.json")
        manifest = [e for e in manifest if e["name"] in a["only"]]
    todo = [e for e in manifest
            if not all(os.path.exists(os.path.join(out, f"{e['name']}.{RUN}.{x}")) for x in ("ids", "txt"))]
    # Check every ids file before the hour-long load, not one by one after it.
    prompts = {e["name"]: checked_ids(set_dir, e) for e in manifest}
    print(f"{len(manifest)} scenarios, {len(manifest) - len(todo)} already done, {len(todo)} to run",
          flush=True)
    if not todo:
        return

    with open(os.path.join(snap, "config.json"), encoding="utf-8") as f:
        mtype = json.load(f).get("model_type", "")
    if a["model"] is not None and mtype not in MODELS[a["model"]]:
        sys.exit(f"--model {a['model']}: {snap} is model_type {mtype!r}, not {' or '.join(MODELS[a['model']])}")
    if mtype in MOE_TYPES:
        moe_main(snap, out, todo, prompts, new_tokens, mtype, a["device"], a["batch"], a["keep"])
        return

    import torch
    from tokenizers import Tokenizer
    from transformers import GenerationConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    spec = importlib.util.spec_from_file_location(
        "b70_check_rotation", os.path.join(HERE, "..", "rotate", "check_rotation.py"))
    CR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CR)

    tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    gen_cfg = GenerationConfig.from_pretrained(snap)
    eos = gen_cfg.eos_token_id
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, eos {eos}", flush=True)

    t = time.time()
    # Spec 14: Agnes 3.0 Flash (config.json model_type agnes) is Qwen3.5 + the parallel
    # FFN under other tensor names (tools/oracle/agnes.py); its bf16 checkpoint
    # (Agnes-AI/Agnes-3.0-Flash) is the A4 reference (spec 14 G3).
    aspec = importlib.util.spec_from_file_location(
        "b70_agnes", os.path.join(HERE, "..", "oracle", "agnes.py"))
    AG = importlib.util.module_from_spec(aspec)
    aspec.loader.exec_module(AG)
    with open(os.path.join(snap, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    if AG.is_agnes(raw):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        kwargs, parallel = AG.translate_text_config(raw)
        tc = Qwen3_5TextConfig(**kwargs)
        tc._attn_implementation = "eager"
    else:
        tc = CR.text_config(snap, 0)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(tc)
        if AG.is_agnes(raw):
            AG.attach_parallel_ffn(model, tc, parallel)
    sd = CR.to_model_names(CR.load_sd(snap, 0, torch.bfloat16))
    if AG.is_agnes(raw):
        sd = {AG.map_name(k): v for k, v in sd.items()}
    want = set(model.state_dict().keys())
    if set(sd) != want:
        sys.exit(f"FATAL: state dict mismatch: {len(want - set(sd))} missing, "
                 f"{len(set(sd) - want)} unexpected, e.g. {sorted(want ^ set(sd))[:4]}")
    model.load_state_dict(sd, strict=True, assign=True)
    del sd
    model.model.rotary_emb = type(model.model.rotary_emb)(tc)
    model.eval()
    print(f"loaded {tc.num_hidden_layers} layers, dtype {model.dtype}, {time.time() - t:.0f}s", flush=True)

    gen_cfg = GenerationConfig(max_new_tokens=new_tokens, do_sample=False, eos_token_id=eos,
                               pad_token_id=eos[0] if isinstance(eos, list) else eos)
    for i, e in enumerate(todo, 1):
        name = e["name"]
        ids = prompts[name]
        t = time.time()
        with torch.no_grad():
            seq = model.generate(torch.tensor([ids]), generation_config=gen_cfg)
        new = [int(x) for x in seq[0, len(ids):].tolist()]
        save(out, name, new, tok.decode(new, skip_special_tokens=False), i, len(todo), len(ids), t)
    print("done", flush=True)


def save(out: str, name: str, new: list[int], text: str, i: int, n: int, n_prompt: int, t: float,
         gaps: list[float] | None = None) -> None:
    # .txt last: its presence (with .ids) marks the scenario done.
    write_atomic(os.path.join(out, f"{name}.{RUN}.ids"), " ".join(map(str, new)) + "\n")
    if gaps is not None:      # the MoE paths: each greedy step's top-1 minus top-2 logit (fp32, exact)
        write_atomic(os.path.join(out, f"{name}.{RUN}.gap"), " ".join("%.9g" % g for g in gaps) + "\n")
    write_atomic(os.path.join(out, f"{name}.{RUN}.txt"), text)
    call = "<tool_call>" in text or "<ifm|tool_call>" in text
    low = f", min gap {min(gaps):.3g}" if gaps else ""
    print(f"[{i}/{n}] {name}: {n_prompt} prompt ids, {len(new)} new, {time.time() - t:.0f}s, call={call}{low}",
          flush=True)


def top2_gap(row) -> float:
    """top-1 minus top-2 of a logits row (0 = a tie: the argmax took the lower id)."""
    import torch
    v = torch.topk(row.float(), 2).values
    return float(v[0] - v[1])


def moe_main(snap: str, out: str, todo: list[dict], prompts: dict, new_tokens: int, mtype: str,
             device: str = "cpu", batch: str = "1", keep: str = "none") -> None:
    import torch
    from tokenizers import Tokenizer
    with open(os.path.join(snap, "generation_config.json"), encoding="utf-8") as f:
        eos = json.load(f)["eos_token_id"]
    eos = set(eos if isinstance(eos, list) else [eos])
    tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, eos {sorted(eos)}, "
          f"model_type {mtype}, {new_tokens} new tokens", flush=True)
    b, kept = 1, (0, 0)
    if batch != "1" or keep != "none":
        with open(os.path.join(snap, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        plan = batch_plan(mtype, cfg, [len(prompts[e["name"]]) for e in todo], new_tokens, mem_limit(), batch, keep)
        print(plan["text"], flush=True)
        b, kept = plan["batch"], (plan["keep_dense"], plan["keep_experts"])
    t = time.time()
    runner = moe_runner(snap, mtype, device, kept)
    print(f"ready in {time.time() - t:.0f}s", flush=True)
    if b == 1:
        print("sequential: one scenario at a time", flush=True)
        for i, e in enumerate(todo, 1):
            name = e["name"]
            ids = prompts[name]
            t = time.time()
            gaps: list[float] = []
            new = runner.generate(ids, new_tokens, eos, on_row=lambda r: gaps.append(top2_gap(r)))
            save(out, name, new, tok.decode(new, skip_special_tokens=False), i, len(todo), len(ids), t, gaps)
        print("done", flush=True)
        return
    print(f"batched: up to {b} scenarios in flight, layer-major (each layer's weights read once per pass, "
          f"every sequence's arithmetic its own: bitwise the sequential run's)", flush=True)
    started: dict[str, float] = {}
    gaps_of: dict[str, list[float]] = {}
    n_done = [0]

    def on_start(name):
        started[name] = time.time()
        gaps_of[name] = []

    def on_done(name, new):
        n_done[0] += 1
        save(out, name, new, tok.decode(new, skip_special_tokens=False), n_done[0], len(todo),
             len(prompts[name]), started[name], gaps_of.pop(name))

    greedy_batched(runner.step_many, runner.new_state, [(e["name"], prompts[e["name"]], new_tokens) for e in todo],
                   b, eos, runner.argmax, on_done, on_row=lambda name, r: gaps_of[name].append(top2_gap(r)),
                   log=lambda s: print(s, flush=True), on_start=on_start)
    print("done", flush=True)


if __name__ == "__main__":
    main()
