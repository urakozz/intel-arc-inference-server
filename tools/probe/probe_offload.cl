// probe_offload.cl - spec 22a (P0.3 / P0.5): the expert-offload probe's helper kernels (tools/probe/probe_offload.cc).
// The expert GEMVs themselves are tools/probe/probe_offload_moe.cl (q4_moe.cl through a pointer table, generated).
//
//   po_page_check(ptrs, first_page, out, pages, ranges)   grid (ceil(pages / 64)), WG 64
//       one lane per 2 MiB page of every range: out[g] = the first u64 of page g - first_page[r] of range r, read
//       through the u64 pointer table `ptrs` [ranges] (first_page [ranges + 1]: the ranges' page prefix sums) -
//       q4_ple_check's form for any number of ranges (21c's is fixed at 32). The host compares with its own view.
//   po_page_tag(ptr, pages, salt)                         grid (ceil(pages / 64)), WG 64
//       the device writes po_tag(page, salt) at the first u64 of every 2 MiB page of one device arena (the alias
//       check: a page another page aliases reads back the later page's tag).
//   po_read_sum(ptr, words, out)                          grid (any), WG 256
//       a wide uint4 read and XOR fold over `words` uint4s from a host or device pointer - bw_sum.cl's pattern: the
//       streaming ceiling of the bytes the expert kernels read.
//   po_stage(route, table, vram_lo, vram_hi, bounce, ltable)   grid (MOE_K, PO_CHUNKS), WG 256
//       the realisable in-list staging (arm K): for routed slot k (route row word k), entry e = table[id] and
//       table[512 + id]; when e lies outside [vram_lo, vram_hi) (a host pointer: a miss) the expert's gate||up and
//       down blocks are copied with uint4 reads into bounce slot k (gate||up at bounce + k x GU_BYTES, down after
//       all MOE_K gate||up slots) and the launch table `ltable` [1024] gets the bounce addresses for id; a hit keeps
//       its table entry. The expert kernels then run bound at ltable / ltable + 512.
//
// Defines: GU_BYTES, DN_BYTES (loader/qwen4exp_layout.h: 1,740,800 / 870,400 B a block), MOE_E (512), MOE_K (10),
// RW (the route row words, 32), PO_CHUNKS (work-groups per slot). Plain OpenCL C 1.2 apart from intptr_t casts.
#ifndef GU_BYTES
#error "probe_offload.cl: GU_BYTES, DN_BYTES, MOE_E, MOE_K, RW, PO_CHUNKS must be defined (tools/probe/CMakeLists.txt)"
#endif
#if (GU_BYTES % 16) != 0 || (DN_BYTES % 16) != 0
#error "probe_offload.cl: the blocks must be whole uint4s"
#endif
#define PO_PAGE ((ulong)2 * 1024 * 1024)

inline ulong po_tag(ulong page, uint salt) { return 0xA11A500000000000UL ^ ((ulong)salt << 32) ^ page; }

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void po_page_check(__global const ulong* restrict ptrs, __global const uint* restrict first_page,
                            __global ulong* restrict out, uint pages, uint ranges) {
  const uint g = get_global_id(0);
  if (g >= pages) return;
  uint lo = 0, hi = ranges;   // the range r with first_page[r] <= g < first_page[r + 1]
  while (hi - lo > 1) {
    const uint mid = (lo + hi) / 2;
    if (first_page[mid] <= g) lo = mid; else hi = mid;
  }
  const ulong addr = ptrs[lo] + (ulong)(g - first_page[lo]) * PO_PAGE;
  out[g] = *(__global const ulong*)(intptr_t)addr;
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void po_page_tag(__global ulong* restrict ptr, uint pages, uint salt) {
  const uint g = get_global_id(0);
  if (g >= pages) return;
  ptr[(size_t)g * (PO_PAGE / 8)] = po_tag(g, salt);
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void po_read_sum(__global const uint4* restrict in, ulong words, __global uint* restrict out) {
  uint acc = 0u;
  for (ulong i = get_global_id(0); i < words; i += get_global_size(0)) {
    const uint4 v = in[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9E3779B9u) out[get_group_id(0)] = acc;   // keeps the loads live, almost never taken
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void po_stage(__global const uint* restrict route, __global const ulong* restrict table, ulong vram_lo,
                       ulong vram_hi, __global uint4* restrict bounce, __global ulong* restrict ltable) {
  const uint k = get_group_id(0);   // the routed slot
  const uint c = get_group_id(1);   // its chunk
  const uint lid = get_local_id(0);
  uint id = route[k];
  id = min(id, (uint)(MOE_E - 1));
  const ulong gu = table[id], dn = table[MOE_E + id];
  const bool miss = gu < vram_lo || gu >= vram_hi;   // the table holds both halves of an expert in one place
  __global uint4* restrict bgu = bounce + (size_t)k * (GU_BYTES / 16);
  __global uint4* restrict bdn = bounce + (size_t)MOE_K * (GU_BYTES / 16) + (size_t)k * (DN_BYTES / 16);
  if (c == 0 && lid == 0) {
    ltable[id] = miss ? (ulong)(intptr_t)bgu : gu;
    ltable[MOE_E + id] = miss ? (ulong)(intptr_t)bdn : dn;
  }
  if (!miss) return;
  const __global uint4* restrict sgu = (const __global uint4*)(intptr_t)gu;
  const __global uint4* restrict sdn = (const __global uint4*)(intptr_t)dn;
  const size_t step = (size_t)PO_CHUNKS * 256;
  for (size_t i = (size_t)c * 256 + lid; i < GU_BYTES / 16; i += step) bgu[i] = sgu[i];
  for (size_t i = (size_t)c * 256 + lid; i < DN_BYTES / 16; i += step) bdn[i] = sdn[i];
}
