// Spec 17a (plan 17a Task 1): the tensor-parallel shapes probe_tp reads (tools/probe/tp_shapes.h), from
// `ModelDesc` through a tp_rank / tp_size view - spec 17 §R2.5. Qwen3.8: 128 all-reduces a token (+ the head's
// exchange), 20 KiB a row; the per-card linears (qkv‖z 5120 x 8192 L1, out_proj / o_proj 3072 x 5120,
// q‖k‖v 5120 x 7168, gate‖up 5120 x 17408, down 8704 x 5120, the int8 head 5120 x 124160); every K slice a whole
// number of 64-groups; the fused layouts re-fused per card from the card's heads. Agnes: 144 + 1, its folded
// gate‖up' 19456 and down' 9728. tp 1 is the one-card table; a MoE model and an uneven split are refused. Host only.
#include <cstdio>
#include <stdexcept>
#include <string>

#include "check.h"
#include "model/model_desc.h"
#include "tp_shapes.h"

namespace {

const tp::Linear& find(const tp::Shapes& s, const std::string& key) {
  for (const tp::Linear& l : s.linears)
    if (key == l.key) return l;
  std::fprintf(stderr, "no linear %s\n", key.c_str());
  std::exit(1);
}

void check_card(const tp::Linear& l, uint32_t K, uint32_t N, uint32_t S, uint32_t L, uint32_t layers) {
  if (l.card.K != K || l.card.N != N || l.card.S != S || l.card.layout != L || l.layers != layers) {
    std::fprintf(stderr, "%s: card %u x %u S%u L%u x%u, want %u x %u S%u L%u x%u\n", l.key, l.card.K, l.card.N,
                 l.card.S, l.card.layout, l.layers, K, N, S, L, layers);
    std::exit(1);
  }
}

template <class F>
bool throws(F f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

void check_qwen38() {
  const model::ModelDesc& d = model::qwen38();
  const tp::Shapes s = tp::tp_shapes(d, 0, 2);
  CHECK_EQ(s.tp_size, 2u);
  CHECK_EQ(s.hidden, 5120u);
  CHECK_EQ(s.calls, 128u);        // 2 a layer x 64
  CHECK_EQ(s.head_exchanges, 1u);
  CHECK_EQ(s.row_bytes, 20480u);  // one fp32 row of the hidden width
  CHECK_EQ(s.token_bytes(1), 128u * 20480u);
  CHECK_EQ(s.token_bytes(8), 8u * 128u * 20480u);
  CHECK_EQ(s.linears.size(), 8u);

  check_card(find(s, "qkvz"), 5120, 8192, 1, 1, 48);
  check_card(find(s, "out"), 3072, 5120, 4, 0, 48);
  check_card(find(s, "qkv"), 5120, 7168, 2, 0, 16);
  check_card(find(s, "o"), 3072, 5120, 4, 0, 16);
  check_card(find(s, "gateup"), 5120, 17408, 8, 0, 64);
  check_card(find(s, "down"), 8704, 5120, 4, 0, 64);
  check_card(find(s, "head"), 5120, 124160, 1, 0, 1);
  check_card(find(s, "ab"), 5120, 128, 1, 0, 48);   // 24 + 24 of 48 + 48, zero-padded to 128

  // the one-card shapes beside them
  const tp::Linear& qkvz = find(s, "qkvz");
  CHECK_EQ(qkvz.one.N, 16384u);
  CHECK_EQ(find(s, "down").one.K, 17408u);
  CHECK_EQ(find(s, "head").one.N, 248320u);
  CHECK_EQ(find(s, "ab").real_n_card, 48u);

  // row-parallel = the all-reduce after it; column-parallel = none
  CHECK(find(s, "out").row_parallel);
  CHECK(find(s, "o").row_parallel);
  CHECK(find(s, "down").row_parallel);
  CHECK(!qkvz.row_parallel);
  CHECK(!find(s, "gateup").row_parallel);
  uint32_t ar = 0;
  for (const tp::Linear& l : s.linears)
    if (l.row_parallel) ar += l.layers;
  CHECK_EQ(ar, s.calls);

  // re-fused per card from the card's heads (never sliced from the one-card buffer)
  CHECK_EQ(s.qkvz_parts.q, 1024u);   // 8 of 16 k-heads x 128
  CHECK_EQ(s.qkvz_parts.k, 1024u);
  CHECK_EQ(s.qkvz_parts.v, 3072u);   // 24 of 48 v-heads x 128
  CHECK_EQ(s.qkvz_parts.z, 3072u);
  CHECK_EQ(s.qkvz_parts.q + s.qkvz_parts.k + s.qkvz_parts.v + s.qkvz_parts.z, qkvz.card.N);
  CHECK_EQ(s.qkv_parts.q, 6144u);    // q‖gate, 12 of 24 heads x 2 x 256
  CHECK_EQ(s.qkv_parts.k, 512u);     // 2 of 4 kv heads
  CHECK_EQ(s.qkv_parts.v, 512u);
  CHECK_EQ(s.qkv_parts.q + s.qkv_parts.k + s.qkv_parts.v, find(s, "qkv").card.N);
  CHECK_EQ(s.conv_channels_card, 5120u);
  CHECK_EQ(s.gdn_v_heads_card, 24u);
  CHECK_EQ(s.fa_q_heads_card, 12u);
  CHECK_EQ(s.fa_kv_heads_card, 2u);
  CHECK_EQ(s.ffn_card, 8704u);       // gate‖up: 8704 + 8704 per card, 16-column interleave blocks
  CHECK_EQ(s.ffn_card % 16, 0u);

  // each K slice whole 64-groups
  for (const tp::Linear& l : s.linears) {
    CHECK_EQ(l.card.K % 64, 0u);
    CHECK_EQ((l.card.K / 64) % l.card.S, 0u);
  }
  CHECK_EQ(find(s, "down").card.K / 64, 136u);
  CHECK_EQ(find(s, "out").card.K / 64, 48u);

  // rank 1 has the same shapes (the split is even)
  const tp::Shapes r1 = tp::tp_shapes(d, 1, 2);
  CHECK_EQ(find(r1, "down").card.K, 8704u);
  CHECK_EQ(r1.vocab_offset, 124160u);
  CHECK_EQ(s.vocab_offset, 0u);
}

void check_agnes() {
  const tp::Shapes s = tp::tp_shapes(model::agnes(), 0, 2);
  CHECK_EQ(s.calls, 144u);
  CHECK_EQ(s.head_exchanges, 1u);
  CHECK_EQ(s.row_bytes, 20480u);
  check_card(find(s, "qkvz"), 5120, 8192, 1, 1, 54);
  check_card(find(s, "out"), 3072, 5120, 4, 0, 54);
  check_card(find(s, "qkv"), 5120, 7168, 2, 0, 18);
  check_card(find(s, "o"), 3072, 5120, 4, 0, 18);
  check_card(find(s, "gateup"), 5120, 19456, 8, 0, 72);
  check_card(find(s, "down"), 9728, 5120, 4, 0, 72);
  CHECK_EQ(find(s, "down").card.K / 64, 152u);
}

void check_tp1_and_refusals() {
  const tp::Shapes s = tp::tp_shapes(model::qwen38(), 0, 1);
  CHECK_EQ(s.tp_size, 1u);
  CHECK_EQ(s.calls, 0u);            // no all-reduce on one card
  CHECK_EQ(s.head_exchanges, 0u);
  const tp::Linear& qkvz = find(s, "qkvz");
  CHECK_EQ(qkvz.card.N, 16384u);
  CHECK_EQ(find(s, "down").card.K, 17408u);
  CHECK_EQ(find(s, "head").card.N, 248320u);
  // a MoE model (Ornith) is not this probe's: dense only
  CHECK(throws([] { tp::tp_shapes(model::ornith(), 0, 2); }));
  // an uneven split (3 does not divide 4 kv heads) and a bad rank
  CHECK(throws([] { tp::tp_shapes(model::qwen38(), 0, 3); }));
  CHECK(throws([] { tp::tp_shapes(model::qwen38(), 2, 2); }));
  CHECK(throws([] { tp::tp_shapes(model::qwen38(), 0, 0); }));
  // by name, as the probe's --model takes it
  CHECK_EQ(&tp::model_by_name("qwen38"), &model::qwen38());
  CHECK_EQ(&tp::model_by_name("agnes"), &model::agnes());
  CHECK(throws([] { tp::model_by_name("bolt"); }));
}

}  // namespace

int main() {
  check_qwen38();
  check_agnes();
  check_tp1_and_refusals();
  std::printf("tp_shapes_test: PASS\n");
  return 0;
}
