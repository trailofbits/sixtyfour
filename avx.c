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
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include "avx.h" 
#include "output.h"

#ifndef __FILE_NAME__
#define __FILE_NAME__ "AVX"
#endif

bool avx2_check(void) {
  return __builtin_cpu_supports("avx2");
}
// use AVX instructions to compare 4 64-bit quantities at once
// The inner loop is unrolled to maximize SIMD execution performance
typedef uint64_t VECTOR[4] __attribute__ ((aligned (32)));

uint64_t avx2_do_range(
    uint64_t start, uint64_t end,
    uint64_t secret, bool *found)
{
  // how many ops were done
  uint64_t out = 0;

  // how many comparisons are done per loop iteration
  const unsigned OPS_PER_LOOP = 32;

  uint64_t max_iterations = (end - start) / OPS_PER_LOOP; 
  //log_output(__FILE_NAME__, "Doing AVX search [%llx, %llx) for [%llx]\n",
  //    start,
  //    end,
  //    secret);

  VECTOR  start_vec  = { start+0, start+1, start+2, start+3 };
  // every iteration, add 4 to each since we compare 4 values at once
  VECTOR  increment  = { 0x4,     0x4,     0x4,     0x4     };

  // Test each lane of the vector against the following values
  VECTOR  secret_vec = { secret,  secret,  secret,  secret  };

  *found = false;

  __m256i avx_start = _mm256_load_si256((__m256i*)&start_vec);
  __m256i avx_increment = _mm256_load_si256((__m256i*)&increment);
  __m256i avx_secret = _mm256_load_si256((__m256i*)&secret_vec);
  uint64_t counter = max_iterations;
  // unroll loop by 8
  while(counter > 0) {
    // compare initial values
    __m256i result0 = _mm256_cmpeq_epi64(avx_start, avx_secret);
    // store comparison results in integer land and not SIMD regs
    int summary0 = _mm256_movemask_epi8(result0);

    // prepare next values for a compare
    __m256i next1 = _mm256_add_epi64(avx_start, avx_increment);
    // do the compare
    __m256i result1 = _mm256_cmpeq_epi64(next1, avx_secret);
    // store results
    int summary1 = _mm256_movemask_epi8(result1);

    // repeat...
    __m256i next2 = _mm256_add_epi64(next1, avx_increment);
    __m256i result2 = _mm256_cmpeq_epi64(next2, avx_secret);
    int summary2 = _mm256_movemask_epi8(result2);

    __m256i next3 = _mm256_add_epi64(next2, avx_increment);
    __m256i result3 = _mm256_cmpeq_epi64(next3, avx_secret);
    int summary3 = _mm256_movemask_epi8(result3);

    __m256i next4 = _mm256_add_epi64(next3, avx_increment);
    __m256i result4 = _mm256_cmpeq_epi64(next4, avx_secret);
    int summary4 = _mm256_movemask_epi8(result4);

    __m256i next5 = _mm256_add_epi64(next4, avx_increment);
    __m256i result5 = _mm256_cmpeq_epi64(next5, avx_secret);
    int summary5 = _mm256_movemask_epi8(result5);

    __m256i next6 = _mm256_add_epi64(next5, avx_increment);
    __m256i result6 = _mm256_cmpeq_epi64(next6, avx_secret);
    int summary6 = _mm256_movemask_epi8(result6);

    __m256i next7 = _mm256_add_epi64(next6, avx_increment);
    __m256i result7 = _mm256_cmpeq_epi64(next7, avx_secret);
    int summary7 = _mm256_movemask_epi8(result7);

    __m256i next8 = _mm256_add_epi64(next7, avx_increment);
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

  uint64_t index = out;
  // check members of the last vector to see which number,
  // if any, was a match
  for(unsigned i = 0; i < OPS_PER_LOOP; i++) {
    uint64_t val = start + (index * OPS_PER_LOOP) + i;
    if(val <= end && val  == secret) {
      *found = true;
      break;
    }
  }

  if(secret == end) {
    *found = true;
  }

  // number of actual comparisons done
  return out*OPS_PER_LOOP;
}

uint64_t avx2_method(uint64_t secret, bool *found, uint64_t h_start, uint64_t h_end) {
  if (!avx2_check()) {
    log_error(__FILE_NAME__,
              "Requested to use AVX2 method but your CPU doesn't support it\n");
    exit(-1);
  }
  return avx2_do_range(h_start, h_end, secret, found);
}
