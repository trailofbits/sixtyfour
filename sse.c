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
#include <inttypes.h>
#include <stdbool.h>
#include <stdlib.h>
#include "output.h"
#include "sse.h"

#ifndef __FILE_NAME__
#define __FILE_NAME__ "SSE"
#endif

#if !defined(__x86_64__)

bool sse_check(void) {
  return false;
}

uint64_t sse_do_range(
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

bool sse_check(void) {
  return __builtin_cpu_supports("sse4.1");
}

// use SSE instructions to compare 2 64-bit quantities at once
// The inner loop is unrolled to maximize SIMD execution performance
typedef uint64_t VECTOR[2] __attribute__ ((aligned (16)));


// SSE 4.1 based loop unrolled to 8 SSE compares (aka 16 ops per loop)
// AVX is better, but this works on older machines
// And was a good test of using intrinsics
uint64_t sse_do_range(
    uint64_t start, uint64_t end,
    uint64_t secret, bool *found)
{
  // how many ops were done
  uint64_t out = 0;

  // how many operations per loop
  const unsigned OPS_PER_LOOP = 16;

  uint64_t max_iterations = (end - start) / OPS_PER_LOOP; 
  // start at the following values
  VECTOR  start_vec  = { start+0, start+1};
  // every iteration, add 4 to each since we compare 4 values at once
  VECTOR  increment  = { 0x2,     0x2};

  // Test each lane of the vector against the following values
  VECTOR  secret_vec = { secret,  secret};

  *found = false;

  __m128i sse_start = _mm_load_si128((__m128i*)&start_vec);
  __m128i sse_increment = _mm_load_si128((__m128i*)&increment);
  __m128i sse_secret = _mm_load_si128((__m128i*)&secret_vec);
  uint64_t counter = max_iterations;
  while(counter > 0) {
    __m128i result0 = _mm_cmpeq_epi64(sse_start, sse_secret);
    int summary0 = _mm_movemask_epi8(result0);

    __m128i next1 = _mm_add_epi64(sse_start, sse_increment);
    __m128i result1 = _mm_cmpeq_epi64(next1, sse_secret);
    int summary1 = _mm_movemask_epi8(result1);

    __m128i next2 = _mm_add_epi64(next1, sse_increment);
    __m128i result2 = _mm_cmpeq_epi64(next2, sse_secret);
    int summary2 = _mm_movemask_epi8(result2);

    __m128i next3 = _mm_add_epi64(next2, sse_increment);
    __m128i result3 = _mm_cmpeq_epi64(next3, sse_secret);
    int summary3 = _mm_movemask_epi8(result3);

    __m128i next4 = _mm_add_epi64(next3, sse_increment);
    __m128i result4 = _mm_cmpeq_epi64(next4, sse_secret);
    int summary4 = _mm_movemask_epi8(result4);

    __m128i next5 = _mm_add_epi64(next4, sse_increment);
    __m128i result5 = _mm_cmpeq_epi64(next5, sse_secret);
    int summary5 = _mm_movemask_epi8(result5);

    __m128i next6 = _mm_add_epi64(next5, sse_increment);
    __m128i result6 = _mm_cmpeq_epi64(next6, sse_secret);
    int summary6 = _mm_movemask_epi8(result6);

    __m128i next7 = _mm_add_epi64(next6, sse_increment);
    __m128i result7 = _mm_cmpeq_epi64(next7, sse_secret);
    int summary7 = _mm_movemask_epi8(result7);

    __m128i next8 = _mm_add_epi64(next7, sse_increment);
    sse_start = next8;

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

uint64_t sse_method(uint64_t secret, bool *found, uint64_t h_start, uint64_t h_end) {
  if (!sse_check()) {
    log_error(
        __FILE_NAME__,
        "Requested to use SSE method but your CPU doesn't support SSE4.1\n");
    exit(-1);
  }
  return sse_do_range(h_start, h_end, secret, found);
} 
