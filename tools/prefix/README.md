# tools/prefix - request logs and the prefix reuse they allow (spec 7 P0)

Record a session:

    build/src/cli/b70-serve <snapshot> --max-len 131072 --log-requests ~/oc-log-1 --port 8000

Each served request becomes `~/oc-log-1/NNNNNN.json` (counted from 1 per server
start): `t_start`, `t_first_token`, `t_end` (steady-clock seconds), `endpoint`,
`request` (the body as received), `prompt_ids`, `out_ids`, `response_text` (every
generated piece, before tool-call parsing).

Analyse it:

    python3 tools/prefix/analyze_log.py ~/oc-log-1

prints, per request, the prompt length, the longest common prefix with the previous
request's ids (prompt + generated) and with any earlier request's, where the best
match ends (an earlier prompt's end, inside earlier generated ids, or elsewhere),
and the prompt tokens a perfect prefix cache could reuse; then the total fraction.

Tests: `python3 tools/prefix/test_analyze_log.py`.
