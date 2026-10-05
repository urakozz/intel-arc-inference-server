// gemv_i8w_run - src/kernels/gemv_i8w.cl on the Mac's OpenCL GPU against a host
// reference (INDICATIVE ONLY: clrun.h says what this can and cannot show).
//
//   gemv_i8w_run [M K N]      defaults 2 1024 512; N % 64 == 0, K % 16 == 0
//
// The host reference is the kernel's formula, not its code: out[m][n] =
// (sum over k ascending of float(q[n][k]) * bf16(x[m][k])) * s[n], fp32 accumulation, with
// the weight in loader/lm_head_int8.h's [n_tile][k16][16 k][16 n] byte layout. The bar is
// a K-scaled relative tolerance, not bit equality: Apple's compiler may contract the
// multiply-add into an fma where ocloc does not (or the reverse), which moves the last
// bits and says nothing about the B70. Bit-exact columns are counted and printed anyway.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "clrun.h"

int main(int argc, char** argv) {
  const int M = argc > 3 ? std::atoi(argv[1]) : 2;
  const int K = argc > 3 ? std::atoi(argv[2]) : 1024;
  const int N = argc > 3 ? std::atoi(argv[3]) : 512;
  if (M < 1 || M > 8 || K <= 0 || K % 16 || N <= 0 || N % 64) {
    std::fprintf(stderr, "gemv_i8w_run: need 1 <= M <= 8, K %% 16 == 0, N %% 64 == 0\n");
    return 2;
  }
  std::mt19937 rng(20261005);
  std::uniform_int_distribution<int> qd(-127, 127);
  std::uniform_real_distribution<float> xd(-1.f, 1.f), sd(0.002f, 0.02f);

  // q[n][k] logically; packed into the kernel's tile layout below.
  std::vector<signed char> q(static_cast<size_t>(N) * K);
  for (auto& v : q) v = static_cast<signed char>(qd(rng));
  std::vector<unsigned char> w(q.size());
  const int K16 = K / 16;
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      w[(static_cast<size_t>(n / 16) * K16 + k / 16) * 256 + (k % 16) * 16 + n % 16] =
          static_cast<unsigned char>(q[static_cast<size_t>(n) * K + k]);
  std::vector<float> s(N);
  for (auto& v : s) v = sd(rng);
  std::vector<std::uint16_t> x(static_cast<size_t>(M) * K);
  for (auto& v : x) v = clrun::f32_to_bf16(xd(rng));

  std::vector<float> ref(static_cast<size_t>(M) * N), mag(ref.size());
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.f, a = 0.f;
      for (int k = 0; k < K; ++k) {
        const float p = static_cast<float>(q[static_cast<size_t>(n) * K + k]) *
                        clrun::bf16_to_f32(x[static_cast<size_t>(m) * K + k]);
        acc += p;
        a += std::fabs(p);
      }
      ref[static_cast<size_t>(m) * N + n] = acc * s[n];
      mag[static_cast<size_t>(m) * N + n] = a * s[n];
    }

  try {
    clrun::Device dev;
    const std::vector<std::string> defs = {"M=" + std::to_string(M), "K=" + std::to_string(K),
                                           "N=" + std::to_string(N)};
    clrun::Program prog(dev, "src/kernels/gemv_i8w.cl", defs);
    clrun::Buffer bw(dev, w), bs(dev, s), bx(dev, x), bo(dev, ref.size() * sizeof(float));
    prog.run("gemv_i8w", {static_cast<size_t>(N)}, {64}, bw, bs, bx, bo);
    const std::vector<float> got = bo.read<float>();

    size_t exact = 0, bad = 0;
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double err = std::fabs(static_cast<double>(got[i]) - ref[i]);
      const double tol = 1e-6 * K * mag[i] + 1e-30;
      if (got[i] == ref[i]) ++exact;
      if (!(err <= tol)) ++bad;
      if (err / tol > worst) worst = err / tol;
    }
    std::printf("gemv_i8w M=%d K=%d N=%d on %s: %zu/%zu bit-exact, worst error %.3g of the "
                "K-scaled tolerance, %zu over\n",
                M, K, N, dev.name().c_str(), exact, got.size(), worst, bad);
    if (bad) {
      std::printf("DISAGREES with the host reference (indicative: read the kernel before the box)\n");
      return 1;
    }
    std::printf("agrees with the host reference (indicative only; the B70 is the real test)\n");
    return 0;
  } catch (const clrun::Error& e) {
    std::fprintf(stderr, "gemv_i8w_run: %s\n", e.what());
    return 1;
  }
}
