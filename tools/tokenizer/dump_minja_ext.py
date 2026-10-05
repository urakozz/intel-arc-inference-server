#!/usr/bin/env python3
"""Expected outputs of the renderer extensions (spec 18d), from Jinja2 itself.

    dump_minja_ext.py

The b70 patches to third_party/minja (third_party/VERSIONS lists them) exist because
K2-Horizon's chat template uses Jinja that minja did not render as Jinja does. Each case
below is a small template exercising one patch; this renders it with Jinja2 configured as
transformers' chat-template renderer is (ImmutableSandboxedEnvironment, trim_blocks,
lstrip_blocks, the tojson filter with ensure_ascii=False) and writes
tests/tokenizer/minja_ext_cases.json, which tests/tokenizer/minja_ext_test.cc renders with
minja and compares. Run in the oracle image:
  docker run --rm -v "$PWD:/ws" -w /ws --entrypoint python3 agnes-ref-img:latest \
      tools/tokenizer/dump_minja_ext.py
"""
import json
import os

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "tokenizer")

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
    x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)

ctx = {"n": None, "t": True, "f": False, "one": 1, "s": "text", "e": {}, "l": [],
       "d": {"k": "v", "n": None, "z": 0}, "pairs": [["a", 1], ["b", 2], ["a", 3]]}

cases = [
    # undefined is not None: `defined` holds for a None value, fails for a missing one.
    ("defined", "{{ n is defined }} {{ d.n is defined }} {{ d.k is defined }} {{ d.missing is defined }} "
                "{{ missing is defined }} {{ d['n'] is not defined }} {{ d.get('missing') is defined }}"),
    # (Printing None itself differs - Jinja writes "None", minja nothing - and is left alone:
    # changing it would move the Qwen3.8 renders; K2's template never prints a bare None.)
    ("defined_none", "{{ n is none }} {{ d.n is none }} {% if d.n is defined %}[{{ d.n is none }}]{% endif %}"
                     "{% set ns = namespace(v=d.missing) %}{{ ns.v is defined }}"),
    # is sameas: True / False / None are singletons.
    ("sameas", "{{ t is sameas true }} {{ f is sameas false }} {{ t is sameas false }} {{ one is sameas true }} "
               "{{ n is sameas none }} {{ s is sameas true }} {{ t is not sameas false }} {{ f is not sameas false }}"),
    ("sameas_if", "{% if t is sameas true %}T{% elif f is sameas false %}F{% else %}-{% endif %}"
                  "{% if f is sameas true %}T{% elif f is sameas false %}F{% endif %}"),
    # str.split() without a separator: whitespace runs, Unicode spaces included, no empty parts.
    ("split", "{{ '  a \t b\n\nc  '.split() | join('|') }}/{{ ''.split() | length }}/"
              "{{ 'x y　z w'.split() | join('|') }}/{{ 'a-b'.split('-') | join('|') }}"),
    # the replace filter, and its precedence over +.
    ("replace", "{{ 'a-b-c' | replace('-', '+') }} {{ 'a-b-c' | replace('-', '+', 1) }} "
                "{{ '[' + 'x\ny' | replace('\n', '\\\\n') + ']' }} {{ 'aaa' | replace('a', 'aa') }}"),
    # dict(): pairs, a mapping, keywords; a repeated key keeps its first position.
    ("dict", "{{ dict(pairs) | tojson }} {{ dict(d) | tojson }} {{ dict(x=1) | tojson }} {{ dict() | tojson }} "
             "{{ dict((d | items | list) + [['w', 9]]) | tojson }}"),
    # attribute getters with an all-digit part index (what `| items` pairs need).
    ("rejectattr_index", "{{ d | items | rejectattr('0', 'equalto', 'n') | list | tojson }} "
                         "{{ d | items | selectattr('0', 'equalto', 'z') | list | tojson }}"),
    ("dict_merge", "{% set m = dict((d | items | list) + (dict(k='w', q=1) | items | rejectattr('0', 'equalto', 'q') | list)) %}"
                   "{{ m | tojson }}"),
    # an empty mapping is falsy.
    ("empty_mapping", "{% if e %}T{% else %}F{% endif %}{% if d %}T{% else %}F{% endif %}"
                      "{{ 'yes' if e else 'no' }}{{ (e or 'fallback') }}"),
]

out = []
for name, source in cases:
    expected = env.from_string(source).render(**ctx)
    out.append({"name": name, "template": source, "expected": expected})
    print(f"{name}: {expected!r}")
spec = {"comment": "tools/tokenizer/dump_minja_ext.py: Jinja2 " + jinja2.__version__ +
                   " renders; minja_ext_test.cc renders each template with minja and compares",
        "context": ctx, "cases": out}
open(os.path.join(here, "minja_ext_cases.json"), "w", encoding="utf-8", newline="\n").write(
    json.dumps(spec, ensure_ascii=False, indent=1) + "\n")
