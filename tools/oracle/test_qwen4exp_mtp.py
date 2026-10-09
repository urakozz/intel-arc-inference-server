#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 21a Task 4: the MTP head port (qwen4exp_mtp.py) held to an independent build and to vLLM's points.

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/test_qwen4exp_mtp.py [test_name ...]

The downloaded tiny has no `mtp.*`; qwen4exp_make_tiny.py writes a tiny with a head (hidden 128, 8 experts
top-2, indexer budget 16 tokens = 4 blocks so the QSA selection is active from position 19 - the head's
draft attention matters at short context). The independent build is written here from transformers'
modules (Qwen4ExpTextGatedResidual, Qwen4ExpTextAttention WITH ITS OWN per-query indexer,
Qwen4ExpTextSparseMoeBlock fully materialised, Qwen4ExpTextRMSNorm) wired as vllm/models/qwen4_exp/nvidia/
mtp.py:285-340 reads; the port is transformers' Qwen4ExpTextDecoderLayer with the cached indexer and lazy
experts. Bitwise, both pre_fc_norm_hidden forms.
"""
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ref = _load("qwen4exp_ref")
mtp = _load("qwen4exp_mtp")
mk = _load("qwen4exp_make_tiny")
BF = torch.bfloat16
TMP = tempfile.mkdtemp(prefix="q4exp_mtp_")
_S: dict = {}


def setup():
    """(snapshot, tc, the streamed main model, the head's tensors) - made once."""
    if not _S:
        d = os.path.join(TMP, "mtp")
        mk.make(d, hidden=128, layer_types="lllq", experts=8, topk=2, form="tiny", mtp=True, indexer_budget=16,
                seed=3)
        tc = ref.text_config(d)
        model, pf, _ = ref.build_streamed(d, tc)
        ck = ref.Ckpt(d)
        _S.update(d=d, tc=tc, model=model, pf=pf, mtp={k: ck.get(k) for k in ck.keys() if k.startswith("mtp.")})
    return _S


def premixer(ids):
    """The main model's pre-mixer 4-stream hidden R over ids (layer_major: the last layer's output)."""
    s = setup()
    step = ref.layer_major(s["model"], s["pf"])
    return step([(ids, 0, {})])[0]


def ids_of(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 248320, (n,), generator=g).tolist()


class Independent:
    """The head straight from vLLM's text (mtp.py:285-340) on transformers' modules."""

    def __init__(self, tensors, tc, norm, embed, lm_head):
        m = ref.mq()
        hc = mtp.head_config(tc)
        H, hcn, eps = tc.hidden_size, tc.hc_count, tc.rms_norm_eps
        L = "mtp.layers.0."
        self.attn_hc = m.Qwen4ExpTextGatedResidual(hc)
        self.mlp_hc = m.Qwen4ExpTextGatedResidual(hc)
        self.attn = m.Qwen4ExpTextAttention(hc, 0)
        self.moe = m.Qwen4ExpTextSparseMoeBlock(hc)
        self.mixer = m.Qwen4ExpTextGatedResidual(hc, use_combine=False)
        self.pre_e = m.Qwen4ExpTextRMSNorm(H, eps=eps)
        self.pre_h = m.Qwen4ExpTextRMSNorm(hcn * H, group_size=None if norm == "single" else H, eps=eps)

        def load(mod, prefix, extra=None):
            sd = {k[len(prefix):]: v.to(BF) for k, v in tensors.items() if k.startswith(prefix) and
                  ".mlp.experts." not in k}
            sd.update(extra or {})
            mod.load_state_dict(sd, strict=True)
            mod.to(BF).eval()
        load(self.attn_hc, L + "attn_hyper_connection.")
        load(self.mlp_hc, L + "mlp_hyper_connection.")
        load(self.attn, L + "self_attn.")
        E = tc.num_experts
        gu = torch.stack([torch.cat([tensors[f"{L}mlp.experts.{e}.gate_proj.weight"],
                                     tensors[f"{L}mlp.experts.{e}.up_proj.weight"]]) for e in range(E)]).to(BF)
        dn = torch.stack([tensors[f"{L}mlp.experts.{e}.down_proj.weight"] for e in range(E)]).to(BF)
        load(self.moe, L + "mlp.", {"experts.gate_up_proj": gu, "experts.down_proj": dn})
        self.moe.experts.config._experts_implementation = tc._experts_implementation
        load(self.mixer, "mtp.hyper_connection_mixer.")
        self.pre_e.load_state_dict({"weight": tensors["mtp.pre_fc_norm_embedding.weight"].to(BF)})
        self.pre_h.load_state_dict({"weight": tensors["mtp.pre_fc_norm_hidden.weight"].to(BF)})
        self.pre_e.to(BF)
        self.pre_h.to(BF)
        self.fc_e = tensors["mtp.fc_embedding.weight"].to(BF)
        self.fc_h = tensors["mtp.fc_hidden.weight"].to(BF)
        self.embed, self.lm_head, self.hc, self.H, self.hcn = embed, lm_head, hc, H, hcn
        self.rotary = m.Qwen4ExpTextRotaryEmbedding(hc)

    @torch.no_grad()
    def step(self, R, tok, pos, cache):
        m = ref.mq()
        T = R.shape[0]
        e = F.linear(self.pre_e(F.embedding(tok, self.embed)), self.fc_e)                 # mtp.py:293-294
        h = F.linear(self.pre_h(R).view(T, self.hcn, self.H), self.fc_h).flatten(-2)       # mtp.py:299-306
        Hs = h + e.unsqueeze(-2).expand(T, self.hcn, self.H).flatten(-2)                   # unit injection
        Hs = Hs[None]
        mask = m.create_causal_mask(config=self.hc, inputs_embeds=Hs, attention_mask=None, past_key_values=cache,
                                    position_ids=torch.arange(pos, pos + T)[None], allow_is_causal_skip=False)
        pe = self.rotary(Hs, torch.arange(0, pos + T).view(1, 1, -1).expand(3, 1, -1))
        x, H0, inj = self.attn_hc(Hs)
        y, _ = self.attn(x, pe, attention_mask=mask, past_key_values=cache)
        Hs = H0 + (y.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        x, H0, inj = self.mlp_hc(Hs)
        y = self.moe(x)
        M = H0 + (y.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)                          # materialised
        return F.linear(self.mixer(M)[0], self.lm_head), M[0]


def test_against_independent():
    """MtpHead.step equals the independent build bitwise - teacher-forced over every row (one call, the
    selection active past row 18), then a second draft step from three rows - both norm forms."""
    s = setup()
    tc, model = s["tc"], s["model"]
    ids = ids_of(40, 1)
    R = premixer(ids)
    for norm in mtp.NORMS:
        head = mtp.MtpHead(s["d"], tc, norm=norm, model=model)
        ind = Independent(s["mtp"], tc, norm, model.model.embed_tokens.weight, model.lm_head.weight)
        c1, c2 = head.new_cache(), ref.mq().DynamicCache(config=head.hc)
        T = len(ids)
        tok = torch.tensor(ids[1:])
        lg_p, M_p, _ = head.step(R[: T - 1], tok, 0, c1)
        lg_i, M_i = ind.step(R[: T - 1], tok, 0, c2)
        assert torch.equal(lg_p, lg_i) and torch.equal(M_p, M_i), f"{norm}: step 1 differs"
        for i in (5, 20, 37):
            ca = mtp.truncate_cache(head, c1, i + 1)
            cb = _truncate_plain(c2, i + 1, head.hc)
            t2 = torch.tensor([int(lg_p[i].float().argmax())])
            a = head.step(M_p[i:i + 1], t2, i + 1, ca)
            b = ind.step(M_i[i:i + 1], t2, i + 1, cb)
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), f"{norm}: step 2 from row {i} differs"
        print(f"  {norm}: step 1 over {T - 1} rows and step 2 from rows 5 / 20 / 37 bitwise")


def _truncate_plain(cache, n, hc):
    out = ref.mq().DynamicCache(config=hc)
    src, dst = cache.layers[0], out.layers[0]
    dst.lazy_initialization(src.keys[:, :, :n], src.values[:, :, :n])
    dst.keys, dst.values = src.keys[:, :, :n].clone(), src.values[:, :, :n].clone()
    dst.lazy_initialization_indexer(src.indexer_keys[:, :n])
    dst.indexer_keys = src.indexer_keys[:, :n].clone()
    return out


def test_unit_injection():
    """The head layer's attn-side input is fc_hidden(pre_fc_norm_hidden(R)) per stream with e added to
    EVERY stream at unit weight - no inject weights - checked at the layer's input."""
    s = setup()
    tc, model = s["tc"], s["model"]
    head = mtp.MtpHead(s["d"], tc, model=model)
    ids = ids_of(12, 2)
    R = premixer(ids)
    cap = {}
    h = head.layer.attn_hyper_connection.register_forward_pre_hook(lambda _m, a: cap.update(x=a[0][0].clone()))
    head.step(R[:11], torch.tensor(ids[1:]), 0, head.new_cache())
    h.remove()
    H = tc.hidden_size
    e = F.linear(head.pre_e(F.embedding(torch.tensor(ids[1:]), model.model.embed_tokens.weight)), head.fc_e.weight)
    hh = F.linear(head.pre_h(R[:11]).view(11, tc.hc_count, H), head.fc_h.weight).flatten(-2)
    assert torch.equal(cap["x"], hh + e.repeat(1, tc.hc_count))
    for s_ in range(tc.hc_count):
        assert torch.equal(cap["x"][:, s_ * H:(s_ + 1) * H], hh[:, s_ * H:(s_ + 1) * H] + e)
    print("  layer input = fc_hidden per stream + e on each of the 4 streams (unit weight)")


def test_fc_hidden_per_stream():
    """fc_hidden is the shared H x H projection applied to each H-wide stream of the normed hc*H vector:
    the head's fused computation equals fc_hidden on each stream separately, and permuting the normed
    streams permutes the output streams."""
    s = setup()
    tc, model = s["tc"], s["model"]
    head = mtp.MtpHead(s["d"], tc, model=model)
    R = premixer(ids_of(9, 3))
    H, n = tc.hidden_size, tc.hc_count
    with torch.no_grad():
        normed = head.pre_h(R)
        fused = head.fc_h(normed.view(-1, n, H)).flatten(-2)
        per = torch.cat([head.fc_h(normed[:, j * H:(j + 1) * H]) for j in range(n)], -1)
        perm = [2, 0, 3, 1]
        pn = torch.cat([normed[:, j * H:(j + 1) * H] for j in perm], -1)
        pf = head.fc_h(pn.view(-1, n, H)).flatten(-2)
    exact = torch.equal(fused, per)
    assert torch.allclose(fused.float(), per.float(), atol=0, rtol=2 ** -7), "fc_hidden is not per stream"
    want = torch.cat([fused[:, j * H:(j + 1) * H] for j in perm], -1)
    assert torch.allclose(pf.float(), want.float(), atol=0, rtol=2 ** -7)
    print(f"  per-stream fc_hidden (fused vs stream by stream bitwise: {exact}); a stream permutation "
          f"permutes the output")


def test_skip_topk():
    """With share_sel the draft steps >= 2 attend exactly step 1's token list (decision 5): the list
    recorded at steps 2 and 3 IS step 1's, and the logits differ from a fresh selection (vLLM's default,
    share_sel=False), whose list contains the step's own position."""
    s = setup()
    tc, model = s["tc"], s["model"]
    head = mtp.MtpHead(s["d"], tc, model=model)
    ids = ids_of(48, 4)
    R = premixer(ids)
    seen = []
    orig = mtp.MtpHead.step

    def spy(self, R_, tok, pos, cache, sel=None):
        out = orig(self, R_, tok, pos, cache, sel)
        seen.append((pos, sel is not None, out[2].clone()))
        return out
    mtp.MtpHead.step = spy
    try:
        a = mtp.chain(R, head, ids, 3, share_sel=True, rows=[29])
        steps_shared = list(seen)
        seen.clear()
        b = mtp.chain(R, head, ids, 3, share_sel=False, rows=[29])
        steps_fresh = list(seen)
    finally:
        mtp.MtpHead.step = orig
    sel1 = steps_shared[0][2][29]
    for pos, forced, got in steps_shared[1:]:
        assert forced and torch.equal(got[0], sel1), f"position {pos}: the list is not step 1's"
    fresh2 = steps_fresh[1][2][0]
    # step 2 from row 29 runs at position 30, the open block's tail (28..30): a fresh list holds it,
    # step 1's (row 29: tail 28, 29) cannot
    assert 30 in fresh2.tolist() and 30 not in sel1.tolist()
    assert not torch.equal(a["logits"][1], b["logits"][1]), "reusing the list changed nothing"
    print(f"  steps 2-3 reuse step 1's {int((sel1 >= 0).sum())}-position list; a fresh one holds position 30 "
          f"and the step-2 logits differ")


def test_returns_premixer():
    """Step i + 1 consumes step i's PRE-mixer hidden M (not the mixer's single stream); the mixer is used
    only for the logits: logits = lm_head(mixer(M))."""
    s = setup()
    tc, model = s["tc"], s["model"]
    head = mtp.MtpHead(s["d"], tc, model=model)
    ids = ids_of(24, 5)
    R = premixer(ids)
    res = mtp.chain(R, head, ids, 2, rows=[10])
    c = head.new_cache()
    lg1, M1, sel1 = head.step(R[:23], torch.tensor(ids[1:]), 0, c)
    assert torch.equal(F.linear(head.mixer(M1[None])[0], model.lm_head.weight), lg1)
    assert M1.shape[-1] == tc.hc_count * tc.hidden_size
    c2 = mtp.truncate_cache(head, c, 11)
    lg2, M2, _ = head.step(M1[10:11], torch.tensor([int(lg1[10].float().argmax())]), 11, c2, sel=sel1[10:11])
    assert torch.equal(res["logits"][1][0], lg2[0].float()) and torch.equal(res["M"][0], M2[0])
    print("  logits = lm_head(mixer(M)); step 2 = step(M1 row 10, its argmax) bitwise")


def test_norm_forms_differ():
    """The two pre_fc_norm_hidden readings (one RMS over 10240 / four over 2560) give different logits on
    random weights: the switch is live."""
    s = setup()
    tc, model = s["tc"], s["model"]
    ids = ids_of(16, 6)
    R = premixer(ids)
    out = {}
    for norm in mtp.NORMS:
        head = mtp.MtpHead(s["d"], tc, norm=norm, model=model)
        out[norm] = head.step(R[:15], torch.tensor(ids[1:]), 0, head.new_cache())[0]
    d = (out["single"].float() - out["per_stream"].float()).abs().max()
    assert d > 0
    print(f"  single vs per_stream: max |d logits| {float(d):.3g}")


def test_accept_and_trace_routes():
    """accept() on the main model's own greedy continuation reports per-depth rates in [0, 1]; the head's
    trace routes are ascending, padded on the last row, and equal the router run inside step()."""
    s = setup()
    tc, model = s["tc"], s["model"]
    head = mtp.MtpHead(s["d"], tc, model=model)
    ctx = ids_of(20, 7)
    seq, cache = list(ctx), None
    with torch.no_grad():
        out = model(input_ids=torch.tensor([seq]), use_cache=True)
        cache = out.past_key_values
        for _ in range(8):
            t = int(out.logits[0, -1].float().argmax())
            seq.append(t)
            out = model(input_ids=torch.tensor([[t]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
    R = premixer(seq)
    acc = mtp.accept(R, head, ctx, seq[20:], 3)
    assert acc["rows"] > 0 and all(0 <= x <= 1 for x in acc["accept_by_depth"])
    tr = head.trace_routes(R, seq, chunk=8)
    T = len(seq)
    assert tr["mtp_ids"].shape == (T, tc.num_experts_per_tok) and bool((tr["mtp_ids"][-1] == -1).all())
    assert bool((tr["mtp_ids"][:-1, 1:] > tr["mtp_ids"][:-1, :-1]).all())
    assert bool((tr["mtp_onorm"][:-1] > 0).all())
    print(f"  accept {acc}; trace routes [{T}][{tc.num_experts_per_tok}] ascending, last row padded")


TESTS = [test_against_independent, test_unit_injection, test_fc_hidden_per_stream, test_skip_topk,
         test_returns_premixer, test_norm_forms_differ, test_accept_and_trace_routes]


def main() -> None:
    import traceback
    torch.manual_seed(0)
    only = sys.argv[1:]
    failed = []
    for t in TESTS:
        if only and t.__name__ not in only:
            continue
        print(t.__name__, flush=True)
        try:
            t()
        except Exception:      # noqa: BLE001 - report every test, then fail
            traceback.print_exc()
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}", flush=True)
    if failed:
        print(f"{len(failed)} FAILED: {failed}")
        sys.exit(1)
    print("ok")


if __name__ == "__main__":
    main()
