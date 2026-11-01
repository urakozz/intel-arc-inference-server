// Spec 26c (host): Bolt's speculative decoding without a device - the verify lists' sizes, launch counts and slots
// (runtime/bolt/bolt_sizes.h), the drafter's prefill tail rule (lili_prefill_r0, Decision 7), the AdaptiveK cost rows
// (server::MtpCost::bolt_verify / bolt_lilicorr - DERIVED, spec 26 §11), b70-serve's --spec refusals and the plan with the
// verify lists and the drafter (cli/bolt_spec.h check_spec / spec_cost, cli/bolt_decode.h settle). The LiLiCorr config is
// tests/model/lilicorr/config.json (the published one).
//
//   1. verify_sizes: the slots 7 x 4 MiB a Mamba block (1,174,405,120 B over the 40 - spec 26 §5.3's 1.17 GB), the union's
//      down rows bf16 [8][9][1024], the scratch at 8 rows; the verify link's landing at 8 rows; the launches - 653 at M 1, 693
//      at M 2..8 with the union MoE (+1 a MoE block: the combine), 653 plain (spec 25 §6.1's list, no fused arm).
//   2. lili_prefill_r0 on the cases Decision 7 names: a chunk ending at a block end (its last 1024 rows), the prompt's last
//      window across chunks, a short prompt, no hook.
//   3. The cost rows: verify 1.000 .. 2.353 (plain) and M 8 1.593 with the union at u 30 (§11), the draft row flat per format
//      (exact 0.500, int8 0.306, int4 0.214), lookup's free drafts, --spec-cost over them, the bf16 head's.
//   4. check_spec: every refusal by name; the formats exact / int8 / int4 and the q forms resolved.
//   5. The plan: two 32.53 GB cards at --max-len auto with the verify lists and the drafter (each format: its weights + buffers
//      at 262144, its prefill staging, the 1.07 GB replica) - the length each allows, printed (DERIVED).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "cli/bolt_decode.h"
#include "cli/bolt_spec.h"
#include "common/json.h"
#include "model/bolt.h"
#include "model/dflash.h"
#include "runtime/bolt/bolt_sizes.h"
#include "runtime/dflash/lilicorr_sizes.h"
#include "server/adaptive_k.h"

namespace {

bool refuses(const std::function<void()>& f, const char* expect) {
  try {
    f();
  } catch (const std::runtime_error& e) {
    if (std::string(e.what()).find(expect) != std::string::npos) return true;
    std::fprintf(stderr, "refused, but the message lacks '%s': %s\n", expect, e.what());
    return false;
  }
  std::fprintf(stderr, "not refused (wanted '%s')\n", expect);
  return false;
}

bool near(double a, double b) { return std::fabs(a - b) < 5e-4; }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bolt_spec_plan_test <lilicorr config.json>\n");
    return 2;
  }
  namespace rb = runtime::bolt;
  const model::BoltDesc& d = model::bolt();
  const model::BoltPlacement two = model::BoltPlacement::two(d, rb::pp_split(d, true));
  const uint32_t ML = 262144;

  // --- 1. the verify lists' sizes, slots and launches ---------------------------------------------------------------------
  {
    size_t slots = 0;
    for (uint32_t dev = 0; dev < 2; ++dev) {
      const rb::VerifySizes v = rb::verify_sizes(d, two, dev, ML, rb::BoltAttn::Flash, rb::BoltState::Fp32);
      uint32_t mamba = 0, moe = 0;
      for (uint32_t l = two.first(dev); l < two.end(dev); ++l) {
        mamba += d.is_mamba(l) ? 1 : 0;
        moe += d.is_moe(l) ? 1 : 0;
      }
      CHECK_EQ(v.slots, size_t(mamba) * 7 * 128 * 64 * 128 * 4);
      CHECK_EQ(v.moe_d, size_t(8) * 9 * 1024 * 2);
      CHECK_EQ(v.scratch.h0, size_t(8) * 4096 * 2);
      CHECK_EQ(v.scratch.logits, size_t(8) * 131072 * 4);
      CHECK_EQ(v.scratch.routes, size_t(88) * 8 * 48 * 4);
      CHECK_EQ(v.scratch.attn_scores, size_t(0));   // flash
      slots += v.slots;
      const rb::VerifySizes b = rb::verify_sizes(d, two, dev, ML, rb::BoltAttn::Flash, rb::BoltState::Bf16);
      CHECK_EQ(b.slots * 2, v.slots);
      std::printf("verify, device %u: scratch %zu B, slots %zu B (%u Mamba blocks), union rows %zu B, link %zu B\n", dev,
                  v.scratch.total(), v.slots, mamba, v.moe_d, rb::verify_link_bytes(d, two, dev));
      (void)moe;
    }
    CHECK_EQ(slots, size_t(1174405120));   // 40 x 7 x 4 MiB: spec 26 §5.3's 1.17 GB (DERIVED)
    const runtime::PpLandingLayout L = rb::verify_landing_layout(d, two);
    CHECK_EQ(L.resid_bytes, size_t(8) * 4096 * 2);
    CHECK_EQ(L.sumsq_bytes, size_t(rb::cut_pending(d, two).total()) * 8 * 4096 * 4);
    const auto U = rb::VerifyMoe::Union, P = rb::VerifyMoe::Plain;
    CHECK_EQ(rb::verify_launches(d, two, 1, rb::BoltAttn::Flash, U), size_t(653));
    CHECK_EQ(rb::decode_launches(d, two, rb::BoltAttn::Flash, rb::BoltFuse{}, runtime::PpHandoff::Copy), size_t(653));
    for (uint32_t M = 2; M <= 8; ++M) {
      CHECK_EQ(rb::verify_launches(d, two, M, rb::BoltAttn::Flash, U), size_t(693));
      CHECK_EQ(rb::verify_launches(d, two, M, rb::BoltAttn::Flash, P), size_t(653));
      CHECK_EQ(rb::verify_launches(d, two, M, rb::BoltAttn::Eager, U), size_t(685));
    }
    bool threw = false;
    try {
      rb::verify_launches(d, two, 9, rb::BoltAttn::Flash, U);   // M 9: the verify lists stop at 8 rows
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
    CHECK(rb::parse_verify_moe("") == U && rb::parse_verify_moe("union") == U && rb::parse_verify_moe("plain") == P);
    CHECK(refuses([&] { rb::parse_verify_moe("fused"); }, "B70_BOLT_VERIFY_MOE=fused"));
    std::printf("verify launches: 653 at M 1, 693 at M 2..8 (union), 653 (plain); slots 1,174,405,120 B over the 40 blocks\n");
  }

  // --- 2. the drafter's prefill tail (Decision 7) ---------------------------------------------------------------------------
  {
    const uint32_t W = 1024, B = 2048;
    CHECK_EQ(rb::lili_prefill_r0(0, 2048, 5000, W, true, B), 1024u);       // ends at the block end 2048: its last 1024 rows
    CHECK_EQ(rb::lili_prefill_r0(2048, 2048, 5000, W, true, B), 1024u);    // 4096
    CHECK_EQ(rb::lili_prefill_r0(4096, 904, 5000, W, true, B), 0u);        // the prompt's end: [3976, 5000) - all of it
    CHECK_EQ(rb::lili_prefill_r0(0, 2048, 4096, W, true, B), 1024u);       // [2048, 4096): the window of the end
    CHECK_EQ(rb::lili_prefill_r0(2048, 2048, 4096, W, true, B), 1024u);
    CHECK_EQ(rb::lili_prefill_r0(0, 300, 300, W, true, B), 0u);            // a short prompt: every row
    CHECK_EQ(rb::lili_prefill_r0(0, 2048, 5000, W, false, B), 2048u);      // no hook: nothing before the end's window
    CHECK_EQ(rb::lili_prefill_r0(2048, 2048, 5000, W, false, B), 1928u);   // rows 3976.. of [2048, 4096)
    CHECK_EQ(rb::lili_prefill_r0(1800, 248, 5000, W, true, B), 0u);        // a short chunk ending at a block end: all of it
    CHECK_EQ(rb::prefill_taps_bytes(W, 6, 4096), size_t(50331648));
    std::printf("prefill tail: block-end chunks project their last 1024 rows, the prompt's end its window; staging 50,331,648 B\n");
  }

  // --- 3. the cost rows (DERIVED, spec 26 §11) -------------------------------------------------------------------------------
  {
    const server::MtpCost p = server::MtpCost::bolt_verify(true, false), u = server::MtpCost::bolt_verify(true, true);
    const double want[8] = {1.000, 1.193, 1.387, 1.580, 1.773, 1.966, 2.160, 2.353};
    CHECK_EQ(p.verify.size(), size_t(8));
    for (int m = 0; m < 8; ++m) CHECK(near(p.verify[m], want[m]));
    CHECK(near(u.verify[0], 1.000) && near(u.verify[7], 1.593));
    for (int m = 1; m < 8; ++m) CHECK(u.verify[m] < p.verify[m] && u.verify[m] > u.verify[m - 1]);
    CHECK_EQ(p.max_k(), 7u);
    const double draft[3] = {0.500, 0.306, 0.214};
    for (unsigned f = 0; f < 3; ++f) {
      const server::MtpCost c = server::MtpCost::bolt_lilicorr(true, f, true);
      CHECK_EQ(c.draft.size(), size_t(7));
      for (double v : c.draft) CHECK(near(v, draft[f]));
      CHECK(near(c.cost(7), 1.593 + draft[f]));
    }
    const server::MtpCost bh = server::MtpCost::bolt_lilicorr(false, 1, true);   // the bf16 head: +0.537 GB a verify / draft
    CHECK(bh.verify[7] < u.verify[7] && bh.draft[0] > 0.306);
    std::printf("cost rows (DERIVED): verify plain");
    for (double v : p.verify) std::printf(" %.3f", v);
    std::printf(" | union");
    for (double v : u.verify) std::printf(" %.3f", v);
    std::printf(" | draft exact 0.500 / int8 0.306 / int4 0.214\n");
  }

  // --- 4. b70-serve's --spec refusals and the flags' resolution (cli/bolt_spec.h) --------------------------------------------
  {
    using cli::bolt::SpecArgs;
    const auto copy = runtime::PpHandoff::Copy, peer = runtime::PpHandoff::Peer;
    {
      SpecArgs s;
      cli::bolt::check_spec(s, 2, copy, true);   // off
      s.spec = "dflash";
      cli::bolt::check_spec(s, 2, copy, true);
      CHECK(s.format == loader::DraftFormat::Int8 && s.q == cli::bolt::BoltDraftQ::Lattice && s.max_k == 7);   // PROVISIONAL
      CHECK(refuses([&] { SpecArgs t = s; cli::bolt::check_spec(t, 2, peer, true); }, "copy hand-off only"));
      cli::bolt::check_spec(s, 1, peer, true);   // one card: no hand-off
    }
    for (const char* f : {"exact", "int8", "int4", "int4g64"}) {
      SpecArgs s;
      s.spec = "dflash";
      s.format_arg = f;
      cli::bolt::check_spec(s, 2, copy, true);
      CHECK(std::string(loader::draft_format_name(s.format)) == (f[0] == 'e' ? "exact" : f[3] == '8' ? "int8" : "int4"));
    }
    {
      SpecArgs s;
      s.spec = "dflash";
      s.format_arg = "bf16";
      CHECK(refuses([&] { cli::bolt::check_spec(s, 2, copy, true); }, "exact, int8 and int4"));
      s.format_arg = "";
      s.q_arg = "lattice+lp";
      cli::bolt::check_spec(s, 2, copy, true);
      CHECK(s.q == cli::bolt::BoltDraftQ::LatticeLp);
      s.q_arg = "raw";
      CHECK(refuses([&] { cli::bolt::check_spec(s, 2, copy, true); }, "lattice or lattice+lp"));
    }
    {
      SpecArgs s;
      s.have_model = true;
      CHECK(refuses([&] { cli::bolt::check_spec(s, 2, copy, true); }, "need --spec dflash"));
      s.have_model = false;
      s.format_arg = "int8";
      s.spec = "lookup";
      CHECK(refuses([&] { cli::bolt::check_spec(s, 2, copy, true); }, "need --spec dflash"));
    }
    {
      SpecArgs s;
      s.have_max_k = true;
      s.max_k = 3;
      CHECK(refuses([&] { cli::bolt::check_spec(s, 2, copy, true); }, "need --spec dflash or --spec lookup"));
      s.spec = "lookup";
      cli::bolt::check_spec(s, 2, copy, true);
      for (uint32_t k : {0u, 8u}) {
        s.max_k = k;
        CHECK(refuses([&] { cli::bolt::check_spec(s, 2, copy, true); }, "expects 1..7"));
      }
    }
    {
      SpecArgs s;
      s.spec = "dflash";
      s.draft_vocab = true;
      CHECK(refuses([&] { cli::bolt::check_spec(s, 2, copy, true); }, "no compact head"));
      s.draft_vocab = false;
      s.mtp_tuning = true;
      CHECK(refuses([&] { cli::bolt::check_spec(s, 2, copy, true); }, "belong to an MTP head"));
    }
    {
      SpecArgs s;
      s.spec = "dflash";
      setenv("B70_PREFILL_REPLAY", "1", 1);
      CHECK(refuses([&] { cli::bolt::check_spec(s, 2, copy, true); }, "unset B70_PREFILL_REPLAY"));
      unsetenv("B70_PREFILL_REPLAY");
    }
    {   // the cost tables
      SpecArgs s;
      s.spec = "dflash";
      s.format = loader::DraftFormat::Int4G64;
      const server::MtpCost c = cli::bolt::spec_cost(s, true, rb::VerifyMoe::Union);
      CHECK(near(c.draft[6], 0.214) && near(c.verify[7], 1.593));
      s.spec = "lookup";
      const server::MtpCost l = cli::bolt::spec_cost(s, true, rb::VerifyMoe::Plain);
      CHECK(near(l.draft[0], 0.0) && near(l.verify[7], 2.353));
      s.spec = "dflash";
      s.cost = "verify=1,1.1,1.2,1.3,1.4,1.5,1.6,1.7;draft=0.2,0.2,0.2,0.2,0.2,0.2,0.2";
      const server::MtpCost o = cli::bolt::spec_cost(s, true, rb::VerifyMoe::Union);
      CHECK(near(o.verify[7], 1.7) && near(o.draft[0], 0.2));
      s.cost = "verify=1,1.1";
      CHECK(refuses([&] { cli::bolt::spec_cost(s, true, rb::VerifyMoe::Union); }, "below --spec-max"));
      s.cost = "nonsense";
      CHECK(refuses([&] { cli::bolt::spec_cost(s, true, rb::VerifyMoe::Union); }, "(--spec-cost)"));
    }
    std::printf("--spec refusals: by name; formats exact / int8 / int4, q lattice / lattice+lp resolved; the cost tables\n");
  }

  // --- 5. the plan with the verify lists and the drafter (DERIVED) ----------------------------------------------------------
  {
    std::ifstream f(argv[1]);
    std::stringstream ss;
    ss << f.rdbuf();
    const model::DFlashDesc dd = model::dflash_desc_from_config(common::json::parse(ss.str()));
    model::check_lilicorr(dd);
    for (loader::DraftFormat fm : {loader::DraftFormat::Exact, loader::DraftFormat::Int8, loader::DraftFormat::Int4G64}) {
      cli::bolt::DecodeArgs a;
      a.max_len = cli::MaxLenArg{true, 0};
      a.prefill = true;
      a.verify_lists = true;
      const rb::BoltReserve r = runtime::dflash::lili_bolt_reserve(dd, d.hidden, d.vocab, fm, d.max_pos);
      a.drafter = r.drafter + rb::prefill_taps_bytes(1024, 6, d.hidden);
      a.drafter_replica = r.drafter_embed_replica;
      CHECK_EQ(a.drafter_replica, size_t(1073741824));
      const rb::BoltPlan pl = cli::bolt::settle(d, a, {cli::bolt::kB70Bytes, cli::bolt::kB70Bytes});
      CHECK(pl.place.devices == 2 && pl.device[1].drafter == a.drafter && pl.device[0].drafter == 0);
      std::printf("plan with --spec dflash, %s drafter: --max-len auto -> %u (DERIVED; device 1 %.3f GB of %.3f)\n",
                  loader::draft_format_name(fm), pl.max_len, double(pl.device[1].total()) / 1e9,
                  double(cli::bolt::kB70Bytes) / 1e9);
    }
  }
  std::printf("bolt_spec_plan_test OK\n");
  return 0;
}
