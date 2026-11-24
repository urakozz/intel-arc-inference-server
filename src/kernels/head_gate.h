// head_gate.h - PERF-123: the head a one-card verify list runs this step, read from runtime::Control::head_cand through
// the extra last argument `gate`. GATE 1: the full int8 head (head_cand 0, sampled requests); GATE 2: the 2-bit
// candidate pass and the exact rescore of its candidates (head_cand 1, greedy requests). Every lane reads the same
// word, so the early return is uniform and comes before any barrier. Without GATE the kernel is unchanged.
#ifndef B70_HEAD_GATE_H
#define B70_HEAD_GATE_H
#ifdef GATE
#define GATE_ARG , __global const uint* restrict gate
#define GATE_RETURN                              \
  if ((gate[0] != 0u) != (GATE == 2)) return;
#else
#define GATE_ARG
#define GATE_RETURN
#endif
#endif
