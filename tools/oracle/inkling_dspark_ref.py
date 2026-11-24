# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""CPU reference of Inkling's recurrent DSpark head.

The head takes draft hidden rows from its caller. The tiny offline exercise
uses fixed synthetic rows built from the mask embedding and the committed
target context, so its acceptance measures only this head fixture.
See vLLM's Inkling2MarkovHead and ops/dspark.py at d23bb0aa50a038f76c23f864733f9729acebca6c.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .inkling_ref import rms, softcap


class DraftHead:
    def __init__(self, snapshot, target):
        cfg = target.config["mtp_config"]
        if cfg["markov_head_type"] != "rnn_hidden":
            raise ValueError("Expected the rnn_hidden DSpark head")
        self.rank = cfg["markov_rank"]
        self.block = cfg["block_size"]
        self.taps = cfg["target_layer_ids"]
        self.eps = cfg["rms_norm_eps"]
        self.limit = target.c["final_logit_softcapping"]
        with safe_open(Path(snapshot) / "mtp.safetensors", framework="pt", device="cpu") as weights:
            self.w = {k.removeprefix("markov_head."): weights.get_tensor(k).float() for k in weights.keys()
                      if k.startswith("markov_head.")}
            self.fuse = weights.get_tensor("fuse_weight")[-1].float().softmax(-1).to(target.dtype)
            self.mask = weights.get_tensor("mask_embed").float()
            self.norm = weights.get_tensor("norm.weight").float()
        self.unembed = target.head
        # The vLLM head folds the shared vocabulary projection into the rank
        # correction, with FP64 accumulation before storing the folded rows.
        projection = self.w["hidden_out.weight"].double()
        bias = self.w["hidden_out.bias"].double()
        self.fold_weight = torch.empty((target.vocab, self.rank), dtype=self.unembed.dtype)
        self.fold_bias = torch.empty(target.vocab, dtype=self.unembed.dtype)
        for start in range(0, target.vocab, 4096):
            rows = slice(start, start + 4096)
            head = self.unembed[rows].double()
            self.fold_weight[rows] = head @ projection
            self.fold_bias[rows] = head @ bias

    def context(self, taps):
        mixed = self.fuse.float() @ taps.float()
        return rms(mixed, self.eps, self.w["dflare_norm.weight"]).to(self.unembed.dtype)

    def step(self, hidden, context, previous, state):
        dtype = self.unembed.dtype
        hidden = hidden.to(dtype)
        sp = torch.cat((state.float(), self.w["markov_w1.weight"][previous]))
        hf = torch.cat((hidden.float(), context.float()))
        recurrent, features = self.w["rnn_proj.weight"].split((sp.numel(), hf.numel()), dim=1)
        raw = F.linear(hf, features, self.w["rnn_proj.bias"]) + F.linear(sp, recurrent)
        gate, candidate, latent = raw.to(dtype).split(self.rank)
        gate = gate.sigmoid()
        state = gate * state.to(dtype) + (1 - gate) * candidate.tanh()
        base = F.linear(hidden, self.unembed)
        correction = F.linear(latent.tanh(), self.fold_weight, self.fold_bias)
        logits = softcap((base + correction).float(), self.limit).to(dtype)
        return logits, state

    def propose(self, hidden_rows, context, anchor):
        state = torch.zeros(self.rank, dtype=self.unembed.dtype)
        prev = anchor
        ids = []
        for row in hidden_rows:
            logits, state = self.step(row, context, prev, state)
            prev = int(logits.argmax())
            ids.append(prev)
        return ids


@torch.inference_mode()
def offline_acceptance(target, head, prompt, rounds):
    if not prompt or rounds < 1:
        raise ValueError("Expected a nonempty prompt and at least one acceptance round")
    if target.config["text_config"]["vocab_size"] > 512:
        raise ValueError("Head-only acceptance requires the tiny synthetic fixture")
    cache = target.cache()
    logits, trace = target.forward(torch.tensor([prompt]), cache)
    anchor = int(logits[0, -1].argmax())
    context = head.context(trace.taps(head.taps)[0, -1])
    mask_rows = torch.arange(1, head.block, dtype=torch.float32)[:, None]
    accepted_lengths = []
    for _ in range(rounds):
        # A fixed synthetic block of post-dense features derived from past-only
        # context. The same proposal is held fixed while the target supplies truth.
        hidden = rms(head.mask + context / mask_rows, head.eps, head.norm)
        proposal = head.propose(hidden, context, anchor)
        matches = 0
        token = anchor
        # Verify the known anchor and the accepted prefix. The first target
        # correction (or the extra greedy token after a full match) is the
        # next anchor, and its row has not been appended to the target cache.
        for guess in [*proposal, None]:
            logits, trace = target.forward(torch.tensor([[token]]), cache)
            token = int(logits[0, -1].argmax())
            if guess != token:
                break
            matches += 1
        accepted_lengths.append(matches)
        anchor = token
        context = head.context(trace.taps(head.taps)[0, -1])
    counts = [sum(n >= i for n in accepted_lengths) for i in range(1, head.block)]
    return {"fixture": "synthetic draft hidden rows", "rounds": rounds, "accepted_lengths": accepted_lengths,
            "accepted_by_position": counts, "acceptance_by_position": [n / rounds for n in counts]}
