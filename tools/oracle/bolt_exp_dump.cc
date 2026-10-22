// Spec 25b: dumps (x, common::exp_torch(x), std::exp(x)) as fp32 triples for every bf16 value x in [-7, 9] - the input of
// tools/oracle/bolt_exp_compare.py (how far the loader's A = -exp(A_log) is from the reference's torch.exp).
//   clang++ -std=c++17 -O2 -ffp-contract=off -Isrc tools/oracle/bolt_exp_dump.cc -o /tmp/bolt_exp_dump
//   /tmp/bolt_exp_dump <out>/exp.bin
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "common/exp_torch.h"
#include "common/bf16.h"
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "wb");
  for (uint32_t bits = 0; bits < 0x10000u; ++bits) {
    float x = common::bf16_to_f32(uint16_t(bits));
    if (!(x >= -7.0f && x <= 9.0f)) continue;
    float y = common::exp_torch(x), z = std::exp(x);
    fwrite(&x, 4, 1, f); fwrite(&y, 4, 1, f); fwrite(&z, 4, 1, f);
  }
  fclose(f);
}
