"""F32-judged expert-set replay at the router's bf16 rounding boundary."""
import math

import torch


def validate_trace(trace, layers, prompt, gen, head, ple, used, vocab):
    s = trace["trace.spec"].tolist()
    if (len(s) != 10 or s[:5] != [1, layers, prompt, gen, 2048] or s[5] < prompt + gen
            or s[5] % 256 or s[6:] != [int(head), int(ple), used, vocab]):
        raise ValueError("engine route capture identity differs from the reference")
    path = trace["trace.path"].tolist()
    if len(path) != 1 or path[0] not in (0, 1):
        raise ValueError("missing engine path identity")
    return path[0]


def judge_route(logits, reference, engine):
    """0 exact, 1 admissible boundary swap, 2 rejected. Window is Ornith's logit bf16 ulp.

    Check the actual exchanged experts, including multiway cut ties. A small adjacent gap must not
    license a distant replacement or dropping an unambiguous high-score expert.
    """
    ref, got = list(map(int, reference)), list(map(int, engine))
    if (len(ref) != len(got) or len(set(ref)) != len(ref) or len(set(got)) != len(got)
            or any(i < 0 or i >= logits.numel() for i in ref + got) or not torch.isfinite(logits).all()):
        raise ValueError("invalid route set or non-finite router logits")
    missing, added = set(ref) - set(got), set(got) - set(ref)
    cut = float(logits[ref].min())
    _, exponent = math.frexp(abs(cut))
    ulp = math.ldexp(1.0, exponent - 8)
    if not missing:
        return 0, 0.0, ulp
    span = float(logits[list(missing)].max()) - float(logits[list(added)].min())
    return (1 if span <= ulp else 2), span, ulp


class RouteReplay:
    def __init__(self, trace, layers, top_k, norm_topk=True):
        self.trace, self.layers, self.k, self.norm = trace, layers, top_k, norm_topk
        self.row0, self.bad = 0, 0
        self.audit, self.hooks = {}, []
        for l in range(layers):
            t = trace[f"route.ids.L{l}"]
            if t.ndim != 2 or t.shape[1] != top_k:
                raise ValueError(f"route.ids.L{l}: expected [rows,{top_k}]")

    def attach(self, model):
        for l, layer in enumerate(model.model.layers):
            self.hooks.append(layer.mlp.gate.register_forward_hook(
                lambda _m, _a, out, l=l: self.apply(l, out)))

    def remove(self):
        for h in self.hooks:
            h.remove()

    def apply(self, layer, out):
        logits, weights, selected = out
        chosen = self.trace[f"route.ids.L{layer}"][self.row0:self.row0 + logits.shape[0]].long()
        if chosen.shape != selected.shape:
            raise ValueError(f"L{layer}: incomplete engine route trace at {self.row0}")
        ids, w = selected.clone(), weights.clone()
        verdicts, spans, ulps, cuts = [], [], [], []
        p = logits.float().softmax(-1)
        for row in range(logits.shape[0]):
            verdict, span, ulp = judge_route(logits[row], selected[row].tolist(), chosen[row].tolist())
            verdicts.append(verdict)
            spans.append(span)
            ulps.append(ulp)
            cuts.append(float(logits[row, selected[row]].min()))
            if verdict == 2:
                self.bad += 1
                if self.bad <= 12:
                    print(f"route REJECT L{layer} row{self.row0+row}: span {span:.9g}, bf16 ulp {ulp:.9g}", flush=True)
                continue
            ids[row] = chosen[row]
            values = p[row].gather(0, chosen[row])
            if self.norm:
                values = values / values.sum()
            w[row] = values.to(w.dtype)
        for name, values, dtype in (("judge", verdicts, torch.int32), ("span", spans, torch.float32),
                                     ("ulp", ulps, torch.float32), ("cut", cuts, torch.float32)):
            self.audit.setdefault(f"route.{name}.L{layer}", []).append(torch.tensor(values, dtype=dtype))
        return logits, w, ids

    def tensors(self):
        return {k: torch.cat(v) for k, v in self.audit.items()}
