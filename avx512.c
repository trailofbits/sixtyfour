/*
 * Copyright (c) 2019 Trail of Bits, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "output.h"
#include "avx512.h" 

#ifndef __FILE_NAME__
#define __FILE_NAME__ "AVX512"
#endif

#if !defined(__x86_64__)
bool avx512_check(void) {
  return false;
}

uint64_t avx512_do_range(
    uint64_t start, uint64_t end,
    uint64_t secret, bool *found)
{
  (void)(start);
  (void)(end);
  (void)(secret);
  *found = false;
  return 0;
}
#else

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

bool avx512_check(void) {
  return __builtin_cpu_supports("avx512bw");
}

// use AVX512 instructions to compare 8 64-bit quantities at once
// The inner loop is unrolled to maximize SIMD execution performance
typedef uint64_t VECTOR[8] __attribute__ ((aligned (32)));

uint64_t avx512_do_range(
    uint64_t start, uint64_t end,
    uint64_t secret, bool *found)
{
  // how many ops were done
  uint64_t out = 0;

  // how many comparison are done per loop
  const unsigned OPS_PER_LOOP = 8*8;

  uint64_t max_iterations = (end - start) / OPS_PER_LOOP; 

  VECTOR  start_vec  = { start+0, start+1, start+2, start+3, start+4, start+5, start+6, start+7 };
  // every iteration, add 8 to each since we compare 8 values at once
  VECTOR  increment  = { 0x8,     0x8,     0x8,     0x8,     0x8,     0x8,     0x8,     0x8     };

  // Test each lane of the vector against the following values
  VECTOR  secret_vec = { secret,  secret,  secret,  secret,  secret,  secret,  secret,  secret  };

  *found = false;

  __m512i avx_start = _mm512_load_si512((__m512i*)&start_vec);
  __m512i avx_increment = _mm512_load_si512((__m512i*)&increment);
  __m512i avx_secret = _mm512_load_si512((__m512i*)&secret_vec);
  uint64_t counter = max_iterations;
  // unroll loop by 8
  while(counter > 0) {
    // compare initial values
    __mmask8 result0 = _mm512_cmpeq_epi64_mask(avx_start, avx_secret);
    // store comparison results in integer land and not SIMD regs
    int summary0 = result0;

    // prepare next values for a compare
    __m512i next1 = _mm512_add_epi64(avx_start, avx_increment);
    // do the compare
    __mmask8 result1 = _mm512_cmpeq_epi64_mask(next1, avx_secret);
    // store results
    int summary1 = result1;

    // repeat...
    __m512i next2 = _mm512_add_epi64(next1, avx_increment);
    __mmask8 result2 = _mm512_cmpeq_epi64_mask(next2, avx_secret);
    int summary2 = result2;

    __m512i next3 = _mm512_add_epi64(next2, avx_increment);
    __mmask8 result3 = _mm512_cmpeq_epi64_mask(next3, avx_secret);
    int summary3 = result3;

    __m512i next4 = _mm512_add_epi64(next3, avx_increment);
    __mmask8 result4 = _mm512_cmpeq_epi64_mask(next4, avx_secret);
    int summary4 = result4;

    __m512i next5 = _mm512_add_epi64(next4, avx_increment);
    __mmask8 result5 = _mm512_cmpeq_epi64_mask(next5, avx_secret);
    int summary5 = result5;

    __m512i next6 = _mm512_add_epi64(next5, avx_increment);
    __mmask8 result6 = _mm512_cmpeq_epi64_mask(next6, avx_secret);
    int summary6 = result6;

    __m512i next7 = _mm512_add_epi64(next6, avx_increment);
    __mmask8 result7 = _mm512_cmpeq_epi64_mask(next7, avx_secret);
    int summary7 = result7;

    __m512i next8 = _mm512_add_epi64(next7, avx_increment);
    avx_start = next8;

    // check whether secret was found in any of the
    // values tested this loop iteration
    if(0 != summary0 ||
       0 != summary1 ||
       0 != summary2 ||
       0 != summary3 ||
       0 != summary4 ||
       0 != summary5 ||
       0 != summary6 ||
       0 != summary7
       )
    {
      // found it
      break;
    }
    --counter;
  }

  out = max_iterations - counter;

  // check members of the last vector to see which number,
  // if any, was a match
  uint64_t index = out;
  for(unsigned i = 0; i < OPS_PER_LOOP; i++) {
    uint64_t val = start + (index * OPS_PER_LOOP) + i;
    if(val <= end && val == secret) {
      *found = true;
      break;
    }
  }

  if(secret == end) {
    *found = true;
  }

  return out*OPS_PER_LOOP;
}
#endif

uint64_t avx512_method(uint64_t secret, bool *found, uint64_t h_start, uint64_t h_end) {
  if (!avx512_check()) {
    log_error(
        __FILE_NAME__,
        "Requested to use AVX512 method but your CPU doesn't support AVX512\n");
    exit(-1);
  }
  return avx512_do_range(h_start, h_end, secret, found);
}
