"""CPU parity of the packed Inkling oracle and the pinned Transformers model."""

import hashlib
from pathlib import Path

import pytest
import torch

from tools.oracle.inkling_ref import Inkling, write_golden
from tools.oracle.dequant import mxfp8_to_f32, nvfp4_to_f32
from tools.oracle.inkling_make_tiny import make_tiny
from tools.oracle.inkling_dspark_ref import DraftHead, offline_acceptance


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.set_num_threads(1)
    path = tmp_path_factory.mktemp("inkling")
    make_tiny(path)
    return path


def makers_model(model):
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
    from transformers.models.inkling2_loop import modeling_inkling2_loop as upstream

    assert hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest() == (
        "1c8e65b45a11c13ecad86c088105acd5e4567123ef6599844a95912d242e9f28")


    config = Inkling2LoopTextConfig.from_dict(model.c, layer_types=[
        "hybrid_sliding" if i in model.c["local_layer_ids"] else "hybrid"
        for i in range(model.c["num_hidden_layers"])
    ])
    # The upstream loader casts checkpoint sources to the requested dtype
    # before its conversion op. Load in FP32 so amax/denominator are preserved,
    # then cast the forward weights using the model's own FP32 exceptions.
    official, loading = upstream.Inkling2LoopForCausalLM.from_pretrained(
        model.snapshot, config=config, dtype=torch.float32, attn_implementation="eager",
        experts_implementation="eager", output_loading_info=True)
    assert not loading["missing_keys"], loading
    assert not loading["mismatched_keys"], loading
    keep_fp32 = official._keep_in_fp32_modules_strict
    for name, value in (*official.named_parameters(), *official.named_buffers()):
        if value.is_floating_point() and not any(key in name for key in keep_fp32):
            value.data = value.data.to(model.dtype)
    return official.eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_makers_parity(tiny, dtype):
    model = Inkling(tiny, dtype)
    official = makers_model(model)
    ids = torch.arange(19)[None] * 7 % 113
    visits = []
    hooks = [layer.register_forward_hook(lambda _m, _a, out: visits.append(out))
             for layer in official.model.layers]
    with torch.inference_mode():
        expected = official(input_ids=ids, use_cache=False).logits
        logits, trace = model.forward(ids)
        compare_visits(trace, visits)
        same(logits, expected, "uncached prompt")
        for prompt in ([ids], [ids[:, :8], ids[:, 8:]]):
            cache = model.cache()
            past = None
            for row in [*prompt, *torch.arange(8).view(8, 1, 1)]:
                expected = official(input_ids=row, cache=past, use_cache=True)
                actual, trace = model.forward(row, cache)
                compare_visits(trace, visits)
                same(actual, expected.logits, "cached logits")
                past = expected.cache
    for hook in hooks:
        hook.remove()


def compare_visits(trace, visits):
    assert len(trace.outputs) == len(visits) == 8
    for i, (actual, expected) in enumerate(zip(trace.outputs, visits)):
        same(actual, expected, f"visit {i}")
    visits.clear()


def same(a, b, what):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), (
        what, (a.float() - b.float()).abs().max().item())


def test_golden_and_head(tiny, tmp_path):
    from safetensors.torch import load_file

    model = Inkling(tiny, torch.float32)
    ids = [3, 7, 11, 19, 31] * 4
    path = tmp_path / "tiny.golden.safetensors"
    write_golden(model, ids, path, gen=8, chunk=8)
    gold = load_file(path)
    assert gold["logits"].shape == (28, 507)
    assert gold["nll"].shape == (19,)
    assert gold["H.L7"].shape == (28, 256)
    assert gold["route.ids.L7"].shape == (28, 4)
    assert gold["tap.3"].shape == (28, 256)
    head = DraftHead(tiny, model)
    result = offline_acceptance(model, head, ids, rounds=2)
    assert result["rounds"] == 2 and len(result["accepted_by_position"]) == 7
    assert sum(result["accepted_by_position"]) == sum(result["accepted_lengths"])


def test_release_expert_formats():
    e2 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    c4 = torch.arange(0, 16, 2, dtype=torch.uint8)
    blocks = (c4 | ((c4 + 1) << 4)).expand(127, -1)
    n = torch.arange(256, dtype=torch.uint8)
    n = n[(n & 127) != 127]
    exponent = ((n.int() >> 3) & 15)
    mantissa = (n & 7).float()
    e4 = torch.ldexp(torch.where(exponent == 0, mantissa, mantissa + 8),
                     torch.where(exponent == 0, -9, exponent - 10))
    e4 = torch.copysign(e4, torch.where(n >= 128, -1., 1.))
    scale4 = n[:127, None].contiguous().view(torch.float8_e4m3fn)
    desired4 = e4[:127, None] * e2
    same(nvfp4_to_f32(blocks, scale4, torch.tensor(3.), torch.tensor(2.)),
         desired4 * 1.5, "NVFP4 codes and global ratio")
    same(desired4, desired4.bfloat16().float(), "NVFP4 local product is exactly BF16")
    values = n[:, None].expand(-1, 32).contiguous().view(torch.float8_e4m3fn)
    exp8 = torch.full((len(n), 1), 123, dtype=torch.uint8)
    same(mxfp8_to_f32(values, exp8), (e4[:, None] / 16).expand(-1, 32), "MXFP8 codes and scale")
    with pytest.raises(ValueError, match="E8M0"):
        mxfp8_to_f32(values, torch.full_like(exp8, 255))


def test_markov_head_recurrence(tiny):
    model = Inkling(tiny, torch.float32)
    head = DraftHead(tiny, model)
    w = head.w
    g = torch.Generator().manual_seed(43)
    rows = torch.randn(7, model.c["hidden_size"], generator=g)
    taps = torch.randn(4, model.c["hidden_size"], generator=g)
    mix = (head.fuse.double()[:, None] * taps.double()).sum(0)
    expected_context = mix * (mix.square().mean() + head.eps).rsqrt() * w["dflare_norm.weight"].double()
    context = head.context(taps)
    torch.testing.assert_close(context, expected_context.float(), atol=1e-6, rtol=1e-6)
    state = torch.zeros(head.rank)
    best = 3
    for h in rows:
        inp = torch.cat([state, w["markov_w1.weight"][best], h, context]).double()
        gate, cand, raw = (w["rnn_proj.weight"].double() @ inp + w["rnn_proj.bias"].double()).chunk(3)
        keep = gate.sigmoid()
        want_state = keep * state.double() + (1 - keep) * cand.tanh()
        hidden = h.double() + w["hidden_out.weight"].double() @ raw.tanh() + w["hidden_out.bias"].double()
        want_logits = head.limit * (model.head.double() @ hidden / head.limit).tanh()
        logits, state = head.step(h, context, best, state)
        torch.testing.assert_close(state, want_state.float(), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(logits, want_logits.float(), atol=2e-6, rtol=1e-6)
        best = int(want_logits.argmax())


def test_teacher_forced_golden(tiny, tmp_path):
    from safetensors.torch import load_file

    model = Inkling(tiny, torch.float32)
    prompt, force = [3, 7, 11], [12, 15]
    path = tmp_path / "forced.golden.safetensors"
    assert write_golden(model, prompt, path, force=force) == force
    gold = load_file(path)
    assert gold["prompt_ids"].tolist() == prompt
    assert gold["tokens"].tolist() == force
    cache = model.cache()
    model.forward(torch.tensor([prompt]), cache)
    row, _ = model.forward(torch.tensor([[force[0]]]), cache)
    same(gold["logits"][len(prompt)], row[0, 0], "forced logit row")
