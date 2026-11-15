# Handoff: Kolibri DSpark (branch `kolibri-dspark`)

Speculative decoding for Kolibri-1 with the community DSpark drafter `intuit/Kolibri-1-dspark`, built blind: box builds
and host tests only, no GPU run. Not merged; a Fable review comes next, then the operator's first card run (after
Flash-Next). Run it with `b70-serve <Kolibri> --spec dflash --draft-model intuit/Kolibri-1-dspark`
(docs/18-running-models.md "Kolibri-1 with the DSpark draft model").

## DSpark against our DFlash (Qwen3.8)

| | DFlash (Qwen3.8) | DSpark (Kolibri-1) |
|---|---|---|
| config | z-lab `dflash_config.target_layer_ids` | speculators: `transformer_layer_config`, `aux_hidden_state_layer_ids` = tap + 1 (taps 3/13/23/33/43) |
| block rows | mask rows 1..K sit AT the position they predict | anchor-first (`sample_from_anchor`): row r predicts p + r + 1, drafts are rows 0..K-1 |
| block attention | non-causal | causal (sliding layers, `sliding_window_non_causal` false) |
| per-row choice | v1 argmax / DFlash 2 selector | Markov head: logit + markov_w2[c] . markov_w1[prev], left to right |
| extras | DFlash 2 convolutions | none (v1 backbone); a confidence head (read for its shape, unused) |
| shapes | hidden 5120, 32/8 heads, FF 17408 | hidden 2560, 20/4 heads, FF 6144, 5 layers, window 2048, RoPE 1e4 |
| target | one engine, verify lists exist | Kolibri: two-card layer split, MoE, no verify lists before this branch |

## Design

- Reused: the DFlash descriptor (a fourth form), DFlashCheckpoint / load_drafter (one more overload: a target by its shape),
  DFlashBuffers and dflash_sizes (the `hid` term), the df_prep / df_attn (CAUSAL=1) / df_topk kernels at DSpark's shapes,
  df_select's DFlash 2 walk with the Markov head as its codebooks and a hidden of ones (the Markov bias is exactly the
  selector's bilinear form), DFlashDraftOut, server::dflash_q / accept_sparse / accept_greedy, Bolt's whole verify and
  drafter contract (enable_spec / attach_drafter / verify / commit / replay_aux, the prefill tap staging, the ring in
  snapshots, the deferred commit in the adapter).
- New: runtime::dflash::DSparkDrafter (its draft and projection lists), Kolibri's verify lists (the decode walk at M rows over
  KolibriVerifyBuffers, kol_attn run in two halves past M 4 over a second Control block, the taps copied on both cards),
  KolibriEngine's spec calls, the planner's SpecReserve, SpecFamily::Kolibri, cli/kolibri_spec.h, the adapter's step_many,
  MtpCost::kolibri_dspark (derived), the CMake block (Kolibri kernels at M 2..8, the drafter's binaries).

## Tests

Host (ran on the box, all pass): dspark_desc_test, dspark_loader_test, dspark_plan_test, dspark_names_test,
kolibri_spec_plan_test, kolibri_spec_test, plus the neighbours (kolibri_plan_test, kolibri_snapshot_test, dflash_loader_test,
lilicorr_*, bolt_spec*, adaptive_k_test, pipeline_args_test).
Card (written, NOT run): kolibri_verify_synth_test, kolibri_verify_synth_pp_test, kolibri_verify_test, kolibri_dspark_test;
the b70-serve refusals cli_reject_serve_kolibri_dspark_* / _dflash_lookup / _lookup (they open Level Zero for the card size,
so they were not run either).

## TODO(card)

1. The cost table (server::MtpCost::kolibri_dspark) is derived from bytes: measure verify M = 1..8 and the draft pass.
2. kol_attn at M 3 / 4 spills (ocloc warning); time the two attention halves.
3. The M-row MoE reads every row's experts: price it, then decide on a union-aware form (Bolt's Decision 10).
4. The taps (`aux - 1`): a low acceptance on the first run points here first.
5. The Markov walk over the head's top 16 against the full vocabulary: the acceptance gap.
6. The prefill tail projects 8 rows a replay (256 replays for a 2048-row window): time it.
7. kolibri_dspark_test's acceptance bar (1.2 tokens a verify) is a placeholder: set the real one from the run.
