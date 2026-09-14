# Spec 4 / Stage 3 - K2-Horizon CPU oracle and golden sets Implementation Plan

**Status (2026-09-14): written, NOT dispatched.** Implementation waits for the operator's ruling on the prefill GEMM direction; these plans may be adjusted after it.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three K2 golden sets (prose, code, cjk × 32 greedy tokens) from the reference `modeling_k2_horizon.py` on CPU. Each set is in the existing golden format plus, for every sparse layer and position, the MoVA and MoE expert ids (ascending) and their bf16 weights, plus the reference's bf16 RoPE rows.

**Architecture:** `tools/oracle/dump_k2.py` is `dump.py`'s shape at K2's scope. It dequantises the checkpoint with `dequant.py`, reads shards through `dump.py`'s `shard_paths()`, builds the model through `trust_remote_code` on the meta device and loads it strictly.
- **Residual taps:** hooks capture them, as in `dump.py`.
- **MoVA routes:** read by wrapping the modeling module's own `calc_router_weights`, which the MoVA attention calls.
- **MoE routes:** recomputed from the router logits the MoE block returns, using the block's own statements.

A `--self-test` mode builds a tiny random K2 and checks both route records against which experts actually ran. The dump itself is operator-gated, since it starts the reference container.

**Tech Stack:** Python 3 in `vllm-xpu-env-next-p314-t215-vxkp0` (torch, transformers ≥ 5.14.1, safetensors), `tools/oracle/run_in_container.sh`, bash.

**Spec:** `docs/superpowers/specs/2026-09-14-spec4-k2-horizon-decode-core-design.md` (§4 "CPU oracle", "Golden gate"; §5 stage 3; §8 risks 2-4)

## Global Constraints

- Branch `spec1.7-codex-exp`; never push. Commits end with `Claude-Session: `.
- **Starting the container is the operator's call. Every container run needs an explicit go, on an otherwise idle box.** CPU only: it can run beside plan 8c's kernel tests, never beside a benchmark.
- Long jobs detached (`setsid nohup … > $HOME/<name>.log 2>&1 < /dev/null &`) and polled; the box's WiFi drops several times an hour.
- Do not edit `tools/oracle/dump.py`, `dequant.py`, `golden.sh`, `run_in_container.sh` or the 27B's golden sets. `dump_k2.py` imports `dump.py`'s helpers by path.
- Prompt ids come from plan 8a's `tests/golden/k2/prompts/{prose,code,cjk}.ids`, which must be committed first.
- **One golden set per checkpoint, in its own directory:** `oracle-out-k2/` (git-ignored like `oracle-out*`; check `.gitignore` covers it, and add the line if not).
- Snapshot passed EXPLICITLY as `c0fd997` inside the container (`/hf/hub/models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ/snapshots/c0fd9971ad6d3bb4caedf726e798a02b8df9bcb5`). `run_in_container.sh`'s `$SNAP` takes the first snapshot directory, and this repo has two; they are identical, but the gate names one.

---

## File structure

| path | responsibility |
|---|---|
| `tools/oracle/dump_k2.py` | the K2 oracle: state dict, strict load, hooks, route records, greedy 32, `--self-test` |
| `tools/oracle/golden_k2.sh` | three container runs, one per prompt, into `oracle-out-k2/` |
| `docs/k2-stage0-facts-2026-09-14.md` (§7) | the dump's measured wall time, peak RSS and token ids |

**Golden file contents** (batch squeezed; `T` prompt ids, `G = 32`):

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `resid.L{i}` / `mixer.L{i}` / `mlp.L{i}` | BF16 | [T, 2560] | as `dump.py`, prompt positions |
| `logits` | F32 | [T + G, 250624] | as `dump.py` |
| `tokens` | I32 | [G] | as `dump.py` |
| `route.mova.ids.L{i}` / `route.moe.ids.L{i}` | I32 | [T + G, 4] / [T + G, 8] | sparse layers 3-47, expert ids **ascending**, every forward (prompt then each generated step) |
| `route.mova.w.L{i}` / `route.moe.w.L{i}` | BF16 | [T + G, 4] / [T + G, 8] | the routing weight each id is multiplied by, same order |
| `rope.cos` / `rope.sin` | BF16 | [T + G, 128] | the reference's cos/sin at every position used |

---

### Task 1: `dump_k2.py`, proven on a tiny random K2

**Files:**
- Create: `tools/oracle/dump_k2.py`, `tools/oracle/golden_k2.sh`
- Modify: `.gitignore` (append `oracle-out-k2/` only if `oracle-out*` does not already match it)

**Interfaces:**
- Consumes: `tools/oracle/dump.py`'s `shard_paths(snapshot)`, `check_quant_config(snapshot)`, `die(msg)`; `tools/oracle/dequant.py`'s `dequant_gptq(qweight, scales, group_size, n_chunk)`; the checkpoint's `modeling_k2_horizon.py` (`calc_router_weights` module function; `K2HorizonSparseMoeBlock.forward` returning `(hidden, router_logits)`; `K2HorizonMoVAAttention`; `K2HorizonRotaryEmbedding`).
- Produces: `dump_k2.py <snapshot> --prompt <ids> --out <file> [--gen 32] [--max-prompt 64]` and `dump_k2.py <snapshot> --self-test`. The golden file layout above is read by plan 8e's `k2_golden_gate_test`.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""The K2-Horizon oracle: golden tensors for one prompt from the reference modeling file, CPU.

    dump_k2.py <snapshot> --prompt <ids-file> --out <out.safetensors> [--gen 32] [--max-prompt 64]
    dump_k2.py <snapshot> --self-test

Spec 4 §4. Same contract as dump.py (read its docstring) plus the routing records and RoPE rows
in plan 8d's table. The weights are dequantised by dequant.py - the function the engine's loader
is bit-compared against - so oracle and engine start from identical bf16 weights. The MoVA
router bias ships F16 and is loaded F16 (assign=True keeps the dtype): the reference widens it
to fp32 before adding it, and so does the engine.

How routes are read, without editing the modeling file:
  * MoVA - K2HorizonMoVAAttention calls the module-level `calc_router_weights`; this script
    replaces that module attribute with a wrapper that records its return value.
  * MoE - K2HorizonSparseMoeBlock computes its routing inline and returns router_logits; a
    forward hook re-runs the block's own selection statements on those logits.
`--self-test` builds a tiny random K2, hooks every expert's forward, and checks both records
against the experts that actually ran - the proof that neither reading drifted from the model.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"oracle_{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dequant = _load("dequant")
_dump = _load("dump")
die, shard_paths, check_quant_config = _dump.die, _dump.shard_paths, _dump.check_quant_config

DEQUANT_CHUNK, DEQUANT_CHUNK_ABOVE = 8192, 65536
QUANT_SUFFIXES = (".scales", ".qzeros", ".g_idx")
F16_ALLOWED = (".self_attn.v_router.bias",)    # the only non-bf16 plain tensor K2 ships


def build_state_dict(snapshot: str, group_size: int) -> dict:
    where = {}
    for path in shard_paths(snapshot):
        h = safe_open(path, framework="pt", device="cpu")
        for key in h.keys():
            if key in where:
                die(f"duplicate tensor name across shards: {key}")
            where[key] = h
    sd, n_quant = {}, 0
    for key in sorted(where):
        if key.endswith(QUANT_SUFFIXES):
            continue
        if key.endswith(".qweight"):
            base = key[: -len(".qweight")]
            qw = where[key].get_tensor(key)
            chunk = DEQUANT_CHUNK if qw.shape[1] > DEQUANT_CHUNK_ABOVE else 0
            w = _dequant.dequant_gptq(qw, where[base + ".scales"].get_tensor(base + ".scales"), group_size, chunk)
            sd[base + ".weight"] = w.t().contiguous()
            n_quant += 1
            del qw, w
        else:
            t = where[key].get_tensor(key)
            if t.dtype == torch.float16 and key.endswith(F16_ALLOWED):
                pass
            elif t.dtype != torch.bfloat16:
                die(f"unquantised tensor {key} is {t.dtype}; K2 ships bf16 except {F16_ALLOWED}")
            sd[key] = t
    print(f"state dict: {len(sd)} tensors, {n_quant} dequantised from int4, "
          f"{sum(t.numel() * t.element_size() for t in sd.values()) / 2**30:.2f} GiB")
    return sd


class Recorder:
    """Routing records keyed by layer, one row appended per forward, ids ascending."""

    def __init__(self, model):
        self.moe, self.mova, self.current = {}, {}, None
        self.handles = []
        modeling = sys.modules[type(model).__module__]
        self._orig = modeling.calc_router_weights
        self._modeling = modeling
        rec = self

        def wrapped(**kwargs):
            w, idx = rec._orig(**kwargs)
            rec._put(rec.mova, rec.current, idx, w)
            return w, idx

        modeling.calc_router_weights = wrapped
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer.self_attn, "v_router"):
                self.handles.append(layer.self_attn.register_forward_pre_hook(self._set_layer(i)))
            if hasattr(layer.mlp, "gate"):
                self.handles.append(layer.mlp.register_forward_hook(self._moe_hook(i)))

    def _set_layer(self, i):
        def fn(_mod, _args):
            self.current = i
        return fn

    def _moe_hook(self, i):
        def fn(mod, _args, out):
            router_logits = out[1]
            # K2HorizonSparseMoeBlock.forward's selection, statement for statement.
            rw = torch.sigmoid(router_logits.to(torch.float32))
            choice = rw + mod.gate.bias.to(rw.dtype)
            _, sel = torch.topk(choice, mod.top_k, dim=-1)
            rw = torch.gather(rw, dim=-1, index=sel)
            if mod.norm_topk_prob:
                rw = rw / rw.sum(dim=-1, keepdim=True)
            rw = (rw * mod.router_scaling_factor).to(torch.bfloat16)
            self._put(self.moe, i, sel, rw)
        return fn

    @staticmethod
    def _put(store, layer, idx, w):
        order = torch.argsort(idx, dim=-1)
        ids = torch.gather(idx, -1, order).to(torch.int32)
        ws = torch.gather(w.to(torch.bfloat16), -1, order)
        store.setdefault(layer, []).append((ids.clone(), ws.clone()))

    def tensors(self) -> dict:
        out = {}
        for name, store in (("moe", self.moe), ("mova", self.mova)):
            for layer, rows in store.items():
                out[f"route.{name}.ids.L{layer}"] = torch.cat([r[0] for r in rows], 0).contiguous()
                out[f"route.{name}.w.L{layer}"] = torch.cat([r[1] for r in rows], 0).contiguous()
        return out

    def close(self):
        self._modeling.calc_router_weights = self._orig
        for h in self.handles:
            h.remove()


def self_test(snapshot: str) -> None:
    """A 4-layer random K2 (1 dense, 3 sparse): both route records == the experts that ran."""
    cfg = AutoConfig.from_pretrained(snapshot, trust_remote_code=True)
    cfg.hidden_size, cfg.num_hidden_layers, cfg.mlp_only_layers = 64, 4, [0]
    cfg.intermediate_size, cfg.moe_intermediate_size = 128, 32
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rope_head_dim = 4, 2, 16, 16
    cfg.vocab_size, cfg.num_experts, cfg.mova_num_experts = 512, 12, 8
    cfg._attn_implementation = "eager"
    torch.manual_seed(7)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(torch.bfloat16).eval()
    with torch.no_grad():
        for layer in model.model.layers:          # make the selection bias actually matter
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.bias.uniform_(-0.2, 0.2)
                layer.self_attn.v_router.bias.uniform_(-0.2, 0.2)
    ran = {}

    def expert_hook(key):
        def fn(_mod, args):
            ran.setdefault(key, 0)
            ran[key] += args[0].shape[0]
        return fn

    hooks = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "experts"):
            for e, ex in enumerate(layer.mlp.experts):
                hooks.append(ex.register_forward_pre_hook(expert_hook(("moe", i, e))))
            for e, ex in enumerate(layer.self_attn.v_experts):
                hooks.append(ex.register_forward_pre_hook(expert_hook(("mova", i, e))))
    rec = Recorder(model)
    ids = torch.randint(0, 512, (1, 9))
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    rec.close()
    for h in hooks:
        h.remove()
    t = rec.tensors()
    for kind, k in (("moe", cfg.num_experts_per_tok), ("mova", cfg.mova_num_experts_per_tok)):
        for layer in (1, 2, 3):
            got = t[f"route.{kind}.ids.L{layer}"]
            assert got.shape == (9, k), (kind, layer, got.shape)
            assert bool((got[:, 1:] > got[:, :-1]).all()), f"{kind} L{layer}: ids not ascending"
            counts = torch.bincount(got.flatten(), minlength=64)
            for e in range(64):
                assert int(counts[e]) == ran.get((kind, layer, e), 0), (kind, layer, e)
    print("dump_k2 self-test OK: MoE and MoVA route records match the experts that ran")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--prompt")
    ap.add_argument("--out")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--max-prompt", type=int, default=64)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check", metavar="GOLDEN", help="print a written golden file's inventory and exit")
    args = ap.parse_args()
    if args.self_test:
        self_test(args.snapshot)
        return
    if args.check:
        with safe_open(args.check, framework="pt", device="cpu") as f:
            keys = sorted(f.keys())
            print(f"{args.check}: {len(keys)} tensors")
            for k in keys:
                if k.endswith((".L3", ".L47")) or ".L" not in k:
                    s = f.get_slice(k)
                    print(f"  {k:<22} {s.get_dtype():<5} {s.get_shape()}")
        return
    if not args.prompt or not args.out:
        die("--prompt and --out are required unless --self-test")

    t0 = time.time()
    torch.manual_seed(0)
    print(f"torch {torch.__version__}, intra-op threads {torch.get_num_threads()}")
    print(f"snapshot {os.path.abspath(args.snapshot)}")
    group_size = check_quant_config(args.snapshot)
    with open(args.prompt, encoding="utf-8") as f:
        ids = [int(x) for x in f.read().split()]
    if not ids or len(ids) > args.max_prompt:
        die(f"prompt has {len(ids)} ids (max {args.max_prompt})")
    print(f"prompt: {len(ids)} ids from {args.prompt}")

    cfg = AutoConfig.from_pretrained(args.snapshot, trust_remote_code=True)
    if cfg.model_type != "k2_horizon":
        die(f"model_type {cfg.model_type!r} is not k2_horizon")
    cfg._attn_implementation = "eager"             # fp32 softmax, spec §3.2's contract
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    sd = build_state_dict(args.snapshot, group_size)
    want = set(model.state_dict().keys())
    missing, unexpected = sorted(want - set(sd)), sorted(set(sd) - want)
    if missing or unexpected:
        die(f"strict load would fail: {len(missing)} missing (first {missing[:3]}), "
            f"{len(unexpected)} unexpected (first {unexpected[:3]})")
    model.load_state_dict(sd, strict=True, assign=True)
    del sd
    model.model.rotary_emb = type(model.model.rotary_emb)(cfg)
    still_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if still_meta:
        die("meta tensors survived the load: " + ", ".join(still_meta[:10]))
    dtypes = {}
    for n, p in model.named_parameters():
        dtypes.setdefault(str(p.dtype), []).append(n)
    print("param dtypes: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(dtypes.items())))
    if len(dtypes.get("torch.float16", [])) != 45:
        die("expected exactly the 45 v_router biases in float16")
    model.eval()
    print(f"loaded strict, {time.time() - t0:.1f}s")

    caps = {}

    def hook(name):
        def fn(_mod, _args, out):
            t = out[0] if isinstance(out, tuple) else out
            caps[name] = t.detach()[0].to(torch.bfloat16).clone()
        return fn

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles += [layer.register_forward_hook(hook(f"resid.L{i}")),
                    layer.self_attn.register_forward_hook(hook(f"mixer.L{i}")),
                    layer.mlp.register_forward_hook(hook(f"mlp.L{i}"))]
    rec = Recorder(model)

    T = len(ids)
    with torch.no_grad():
        t_fwd = time.time()
        out = model(input_ids=torch.tensor([ids]), use_cache=True)
        for h in handles:
            h.remove()
        print(f"prompt forward: {time.time() - t_fwd:.1f}s")
        cache = out.past_key_values
        rows = [out.logits[0].to(torch.float32)]
        tokens = []
        t_gen = time.time()
        for _ in range(args.gen):
            nxt = int(torch.argmax(rows[-1][-1]).item())
            tokens.append(nxt)
            out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            rows.append(out.logits[0].to(torch.float32))
        print(f"greedy {args.gen}: {time.time() - t_gen:.1f}s -> {tokens}")
        cos, sin = model.model.rotary_emb(torch.zeros(1, 1, dtype=torch.bfloat16),
                                          torch.arange(T + args.gen).unsqueeze(0))
    rec.close()

    tensors = dict(caps)
    tensors.update(rec.tensors())
    tensors["logits"] = torch.cat(rows, 0).contiguous()
    tensors["tokens"] = torch.tensor(tokens, dtype=torch.int32)
    tensors["rope.cos"] = cos[0].contiguous()
    tensors["rope.sin"] = sin[0].contiguous()

    n_layers = cfg.num_hidden_layers
    if len(caps) != 3 * n_layers:
        die(f"captured {len(caps)} activations, expected {3 * n_layers}")
    sparse = [i for i in range(n_layers) if i not in cfg.mlp_only_layers]
    for kind, k in (("moe", cfg.num_experts_per_tok), ("mova", cfg.mova_num_experts_per_tok)):
        for i in sparse:
            got = tensors.get(f"route.{kind}.ids.L{i}")
            if got is None or tuple(got.shape) != (T + args.gen, k):
                die(f"route.{kind}.ids.L{i}: {None if got is None else tuple(got.shape)}, "
                    f"expected ({T + args.gen}, {k})")
    if tensors["logits"].shape[0] != T + args.gen:
        die(f"logits rows {tensors['logits'].shape[0]} != {T + args.gen}")
    if not torch.isfinite(tensors[f"resid.L{n_layers - 1}"].to(torch.float32)).all():
        die("last residual has NaN/Inf - the dequantised weights are wrong")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(tensors, args.out, metadata={
        "snapshot": os.path.abspath(args.snapshot), "prompt_ids": " ".join(map(str, ids)),
        "n_prompt": str(T), "gen": str(args.gen), "attn_implementation": "eager",
        "group_size": str(group_size), "model_type": "k2_horizon"})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"wrote {args.out}: {len(tensors)} tensors, {total / 2**20:.1f} MiB")
    print(f"wall {time.time() - t0:.1f}s, peak RSS {rss:.1f} GiB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `tools/oracle/golden_k2.sh`**

```bash
#!/usr/bin/env bash
# The three K2-Horizon golden sets (plan 8d), one container run per prompt, ON THE BOX:
#   OUT_DIR=oracle-out-k2 setsid nohup tools/oracle/golden_k2.sh > oracle-out-k2/golden.log 2>&1 </dev/null &
# The snapshot is named explicitly (run_in_container.sh's $SNAP would take the first of this
# repo's two identical snapshots). PROMPTS selects which sets to build (default all three).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT_DIR="${OUT_DIR:-oracle-out-k2}"
IDS_DIR="${IDS_DIR:-tests/golden/k2/prompts}"
K2_REPO="models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ"
K2_SNAP="/hf/hub/$K2_REPO/snapshots/c0fd9971ad6d3bb4caedf726e798a02b8df9bcb5"
mkdir -p "$OUT_DIR"
echo "=== K2 golden set -> $OUT_DIR   ids <- $IDS_DIR   prompts: ${PROMPTS:-prose code cjk}"
for p in ${PROMPTS:-prose code cjk}; do
  echo "=== $p  start $(date -Is)"
  rc=0
  ORACLE_MODEL="$K2_REPO" tools/oracle/run_in_container.sh \
    "python3 tools/oracle/dump_k2.py $K2_SNAP --prompt /ws/$IDS_DIR/$p.ids \
       --out /ws/$OUT_DIR/$p.golden.safetensors --gen 32 --max-prompt 64" \
    > "$OUT_DIR/$p.log" 2>&1 || rc=$?
  echo "=== $p  done  $(date -Is)  rc=$rc"
  tail -3 "$OUT_DIR/$p.log"
  [ "$rc" -eq 0 ] || exit "$rc"
done
echo "=== ALL DONE $(date -Is)"
```

`chmod +x tools/oracle/golden_k2.sh tools/oracle/dump_k2.py`. Then check `git check-ignore -v oracle-out-k2/x` prints an ignore rule; if it prints nothing, append `oracle-out-k2/` to `.gitignore`.

- [ ] **Step 3: Commit before any container run**

```bash
git add tools/oracle/dump_k2.py tools/oracle/golden_k2.sh .gitignore
git commit -m "tools(oracle): dump_k2.py - K2 golden tensors with MoE/MoVA route records, self-test

Claude-Session: "
```

- [ ] **Step 4: On the operator's go, run the self-test** (seconds, no weights; the HF cache mount only serves config.json and the remote code):
  `tools/box.sh sync && ssh -o BatchMode=yes user@box 'cd ~/b70-inference-server && ORACLE_MODEL=models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ setsid nohup tools/oracle/run_in_container.sh "python3 tools/oracle/dump_k2.py /hf/hub/models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ/snapshots/c0fd9971ad6d3bb4caedf726e798a02b8df9bcb5 --self-test" > $HOME/k2-selftest.log 2>&1 < /dev/null &'`
  Poll `tail -5 $HOME/k2-selftest.log`. Expected: `dump_k2 self-test OK`. **An assertion is a defect in the recorder - fix it here, before any 75 GB load.** If `calc_router_weights` is not a module attribute in this image's copy of the remote code (an `AttributeError` at `Recorder.__init__`), the MoVA reading is wrong by construction: stop and report.

---

### Task 2: The three golden sets - OPERATOR-GATED

**Files:**
- Modify: `docs/k2-stage0-facts-2026-09-14.md` (§7)

**Interfaces:**
- Consumes: Task 1's scripts, plan 8a's prompt ids.
- Produces: `oracle-out-k2/{prose,code,cjk}.golden.safetensors` on the box. Plan 8e's gate test reads them from `${CMAKE_SOURCE_DIR}/oracle-out-k2`.

- [ ] **Step 1: Prove the box idle for CPU work and ask for the go** - `ssh -o BatchMode=yes user@box 'docker ps --format "{{.Names}}" ; uptime; free -g | head -2'`. It needs no other container and at least 95 GB available (spec §4 estimates a ~87 GB peak). Report this to the operator and wait for an explicit go.

- [ ] **Step 2: Run `prose` alone first** - it measures the real peak before three runs commit to it:
  `ssh -o BatchMode=yes user@box 'cd ~/b70-inference-server && mkdir -p oracle-out-k2 && PROMPTS=prose setsid nohup tools/oracle/golden_k2.sh > oracle-out-k2/golden-prose.log 2>&1 < /dev/null &'`
  Poll every few minutes: `ssh -o BatchMode=yes user@box 'tail -3 ~/b70-inference-server/oracle-out-k2/golden-prose.log; tail -2 ~/b70-inference-server/oracle-out-k2/prose.log; free -g | sed -n 2p'`.
  Expected, in `prose.log`: `state dict: … GiB`, `param dtypes: torch.bfloat16=…, torch.float16=45`, `loaded strict`, `greedy 32: … -> [...]`, `wrote …`, `wall …, peak RSS … GiB`.
  **If it is killed for memory (rc 137):** record the last `free -g` line and stop. The fix (a bigger chunk-at-a-time dequant, or streaming shards) is a ruling, not a retry.

- [ ] **Step 3: Check the file** (in the container, seconds):
  `ssh -o BatchMode=yes user@box 'cd ~/b70-inference-server && ORACLE_MODEL=models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ tools/oracle/run_in_container.sh "python3 tools/oracle/dump_k2.py x --check oracle-out-k2/prose.golden.safetensors"'`
  Expected: `328 tensors` (3 × 48 taps + 4 × 45 route tensors + `logits`, `tokens`, `rope.cos`, `rope.sin`). `route.moe.ids.L47` is I32 `[T+32, 8]`; `rope.cos` is BF16 `[T+32, 128]`.

- [ ] **Step 4: Run `code` and `cjk`** - `PROMPTS="code cjk"`, the same command as Step 2 with the log `golden-rest.log`, then the same check for both files.

- [ ] **Step 5: Record** - add §7 to the facts doc:

```markdown
## 7. K2 golden sets (dump_k2.py, plan 8d) - measured

| prompt | T | wall (s) | peak RSS (GiB) | tokens[0..7] |
|---|---:|---:|---:|---|
| prose | <T> | <wall> | <rss> | <ids> |
| code | … | … | … | … |
| cjk | … | … | … | … |

Image `vllm-xpu-env-next-p314-t215-vxkp0`, transformers <version from the log>, snapshot c0fd997,
eager attention, 328 tensors per file. Spec §4's ~87 GB peak estimate vs measured: <measured>.
```

Commit: `git add docs/k2-stage0-facts-2026-09-14.md && git commit -m "docs(k2): the three K2 golden sets - wall, peak RSS, tokens" -m "Claude-Session: "`.

---

## Self-review

**Spec coverage** (§4 "CPU oracle"):
- the reference modeling file via `trust_remote_code` → Task 1;
- `dequant.py` weights and `shard_paths()` → Task 1;
- the existing golden format plus per-sparse-layer, per-position MoE and MoVA ids (ascending) and weights → Task 1's `Recorder`, proven by `--self-test`;
- peak memory measured → Task 2 Step 2, prose first;
- container only on the operator's go → Tasks 1 Step 4 and 2.

§8 risk 2 (top-k tie order) is read from torch's own `topk` in both recorders, so the golden ids ARE torch's tie choice.

**Additions:**
- `rope.cos` and `rope.sin` let plan 8e diagnose plan 8b's fp32 RoPE construction against the reference.
- Route rows cover generated steps as well as the prompt, so a routing divergence after the first teacher-forced step is still localisable.

**Placeholders:** the `<…>` in Step 5 are measured outputs.

**Names:** `route.{moe,mova}.{ids,w}.L{i}` and `rope.{cos,sin}` are the tensor names plan 8e reads.
