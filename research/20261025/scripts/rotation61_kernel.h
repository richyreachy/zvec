#pragma once
#include <immintrin.h>
#include <cstddef>
struct alignas(64) Rotation61 {
  struct alignas(64) Layer {
    float c[384], s[384];
  };
  Layer layers[3];
  __attribute__((target("avx512f,fma"), noinline)) void apply(float *v) const {
    const __m512i even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18,
                                           20, 22, 24, 26, 28, 30);
    const __m512i odd = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21,
                                          23, 25, 27, 29, 31);
    const __m512i low = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5,
                                          21, 6, 22, 7, 23);
    const __m512i high = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28,
                                           13, 29, 14, 30, 15, 31);
    for (size_t base = 0; base < 768; base += 32) {
      __m512 p = _mm512_loadu_ps(v + base), q = _mm512_loadu_ps(v + base + 16);
      __m512 a = _mm512_permutex2var_ps(p, even, q),
             b = _mm512_permutex2var_ps(p, odd, q);
      __m512 c = _mm512_loadu_ps(layers[0].c + base / 2),
             s = _mm512_loadu_ps(layers[0].s + base / 2);
      __m512 x = _mm512_fmsub_ps(a, c, _mm512_mul_ps(b, s)),
             y = _mm512_fmadd_ps(b, c, _mm512_mul_ps(a, s));
      _mm512_storeu_ps(v + base, _mm512_permutex2var_ps(x, low, y));
      _mm512_storeu_ps(v + base + 16, _mm512_permutex2var_ps(x, high, y));
    }
    for (size_t base = 0; base < 768; base += 32) {
      __m512 a = _mm512_loadu_ps(v + base), b = _mm512_loadu_ps(v + base + 16),
             c = _mm512_loadu_ps(layers[1].c + base / 2),
             s = _mm512_loadu_ps(layers[1].s + base / 2);
      _mm512_storeu_ps(v + base, _mm512_fmsub_ps(a, c, _mm512_mul_ps(b, s)));
      _mm512_storeu_ps(v + base + 16,
                       _mm512_fmadd_ps(b, c, _mm512_mul_ps(a, s)));
    }
    for (size_t base = 0; base < 384; base += 16) {
      __m512 a = _mm512_loadu_ps(v + base), b = _mm512_loadu_ps(v + base + 384),
             c = _mm512_loadu_ps(layers[2].c + base),
             s = _mm512_loadu_ps(layers[2].s + base);
      _mm512_storeu_ps(v + base, _mm512_fmsub_ps(a, c, _mm512_mul_ps(b, s)));
      _mm512_storeu_ps(v + base + 384,
                       _mm512_fmadd_ps(b, c, _mm512_mul_ps(a, s)));
    }
  }
};
