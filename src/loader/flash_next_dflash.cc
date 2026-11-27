#include "loader/flash_next_dflash.h"

#include "loader/snapshot.h"

// flash_next_dflash.cc - the host half of the Flash-Next DFlash 2 drafter's loader arm (flash_next_dflash.h): the
// generic DFlash checkpoint opened at Flash-Next's shape, held to its kernels and its target, and the load format
// resolved. No Level Zero: tests/loader/flash_next_dflash_loader_test.cc runs it on any host.
namespace loader::flash_next {

DFlashCheckpoint open(const std::string& snapshot_or_repo, DraftFormat& format, bool require_packed_g64) {
  const std::string dir = resolve_snapshot(snapshot_or_repo);
  const auto d = model::dflash_desc_from_config(read_json(dir + "config.json"));
  DFlashCheckpoint ck(d, dir, require_packed_g64);
  format = ck.packed() ? DraftFormat::Int4G64 : DraftFormat::Int8;
  return ck;
}

void check(const model::DFlashDesc& d, const model::DFlashTargetShape& target) {
  model::check_flash_next_dflash(d);
  model::check_dflash_target(d, target);
}

void check(const DFlashCheckpoint& ck, const model::DFlashTargetShape& target) {
  check(ck.desc(), target);
  ck.check_target(target);
}

std::vector<std::pair<uint32_t, uint32_t>> linear_shapes(const model::DFlashDesc& d, uint32_t target_hidden) {
  std::vector<std::pair<uint32_t, uint32_t>> v;
  v.emplace_back(d.fc_k(target_hidden), d.hidden);                           // fc
  for (uint32_t l = 0; l < d.layers; ++l) {
    v.emplace_back(d.hidden, d.qkv_n());                                      // q||k||v
    v.emplace_back(d.q_n(), d.hidden);                                        // o
    v.emplace_back(d.hidden, d.gate_up_n());                                  // gate||up
    v.emplace_back(d.inter, d.hidden);                                        // down
    if (d.v2()) {
      v.emplace_back(d.hidden, d.kproj_n());                                  // attention_conv's
      v.emplace_back(d.hidden, d.kproj_n());                                  // mlp_conv's
    }
  }
  if (d.v2()) v.emplace_back(d.hidden, d.sel_rank);                           // the selector's
  v.emplace_back(d.hidden, 2 * d.kv_n());                                     // the context k||v (qkv's rows)
  return v;
}

}  // namespace loader::flash_next
