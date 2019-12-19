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
#include <stdlib.h>
#include "neon.h" 
#include "output.h"

#ifndef __FILE_NAME__
#define __FILE_NAME__ "NEON"
#endif

#if !defined(__ARM_NEON)
bool neon_check(void) {
  return false;
}

uint64_t neon_do_range(
    uint64_t start, uint64_t end,
    uint64_t secret, bool *found)
{
  (void)start;
  (void)end;
  (void)secret;
  *found = false;
  return 0;
}
#else

#include <arm_neon.h>

bool neon_check(void) {
  return true;
}

uint64_t neon_do_range(
    uint64_t start, uint64_t end,
    uint64_t secret, bool *found)
{
  uint64x2_t start_reg = {start+0, start+1};
  uint64x2_t secret_reg = {secret, secret};
  uint64x2_t increment = {0x2, 0x2};

  uint64_t out = 0;
  const unsigned OPS_PER_LOOP = 32;
  uint64_t max_iterations = (end - start) / OPS_PER_LOOP; 

  uint64_t counter = max_iterations;
  uint64x2_t i0 = start_reg;
  uint64x2_t i1 = vaddq_u64(i0, increment);
  uint64x2_t i2 = vaddq_u64(i1, increment);
  uint64x2_t i3 = vaddq_u64(i2, increment);
  uint64x2_t i4 = vaddq_u64(i3, increment);
  uint64x2_t i5 = vaddq_u64(i4, increment);
  uint64x2_t i6 = vaddq_u64(i5, increment);
  uint64x2_t i7 = vaddq_u64(i6, increment);
  uint64x2_t i8 = vaddq_u64(i7, increment);
  uint64x2_t i9 = vaddq_u64(i8, increment);
  uint64x2_t ia = vaddq_u64(i9, increment);
  uint64x2_t ib = vaddq_u64(ia, increment);
  uint64x2_t ic = vaddq_u64(ib, increment);
  uint64x2_t id = vaddq_u64(ic, increment);
  uint64x2_t ie = vaddq_u64(id, increment);
  uint64x2_t iF = vaddq_u64(ie, increment);
  uint64x2_t next = vaddq_u64(iF, increment);

  while(counter > 0) {
    // compare i0, secret 
    i0 = vceqq_u64(i0, secret_reg);
    i1 = vceqq_u64(i1, secret_reg);
    i2 = vceqq_u64(i2, secret_reg);
    i3 = vceqq_u64(i3, secret_reg);

    i4 = vceqq_u64(i4, secret_reg);
    i5 = vceqq_u64(i5, secret_reg);
    i6 = vceqq_u64(i6, secret_reg);
    i7 = vceqq_u64(i7, secret_reg);

    i8 = vceqq_u64(i8, secret_reg);
    i9 = vceqq_u64(i9, secret_reg);
    ia = vceqq_u64(ia, secret_reg);
    ib = vceqq_u64(ib, secret_reg);

    ic = vceqq_u64(ic, secret_reg);
    id = vceqq_u64(id, secret_reg);
    ie = vceqq_u64(ie, secret_reg);
    iF = vceqq_u64(iF, secret_reg);

    uint64x2_t r0 = vorrq_u64(i0, i1);
    uint64x2_t r1 = vorrq_u64(i2, i3);
    uint64x2_t r1_l1 = vorrq_u64(r0, r1);

    uint64x2_t r2 = vorrq_u64(i4, i5);
    uint64x2_t r3 = vorrq_u64(i6, i7);
    uint64x2_t r2_l1 = vorrq_u64(r2, r3);

    uint64x2_t r4 = vorrq_u64(i8, i9);
    uint64x2_t r5 = vorrq_u64(ia, ib);
    uint64x2_t r3_l1 = vorrq_u64(r4, r5);

    uint64x2_t r6 = vorrq_u64(ic, id);
    uint64x2_t r7 = vorrq_u64(ie, iF);
    uint64x2_t r4_l1 = vorrq_u64(r6, r7);

    uint64x2_t r1_l2 = vorrq_u64(r1_l1, r2_l1);
    uint64x2_t r2_l2 = vorrq_u64(r3_l1, r4_l1);

    uint64x2_t res = vorrq_u64(r1_l2, r2_l2);

    if (vgetq_lane_u64(res, 0) != vgetq_lane_u64(res, 1))
    {
      break;
    }
    --counter;

    i0 = next;
    i1 = vaddq_u64(i0, increment);
    i2 = vaddq_u64(i1, increment);
    i3 = vaddq_u64(i2, increment);
    i4 = vaddq_u64(i3, increment);
    i5 = vaddq_u64(i4, increment);
    i6 = vaddq_u64(i5, increment);
    i7 = vaddq_u64(i6, increment);

    i8 = vaddq_u64(i7, increment);
    i9 = vaddq_u64(i8, increment);
    ia = vaddq_u64(i9, increment);
    ib = vaddq_u64(ia, increment);
    ic = vaddq_u64(ib, increment);
    id = vaddq_u64(ic, increment);
    ie = vaddq_u64(id, increment);
    iF = vaddq_u64(ie, increment);
    next = vaddq_u64(iF, increment);
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


uint64_t neon_method(uint64_t secret, bool *found, uint64_t h_start, uint64_t h_end) {
  if (!neon_check()) {
    log_error(__FILE_NAME__,
              "Requested to use neon method but your CPU doesn't support it\n");
    exit(-1);
  }
  return neon_do_range(h_start, h_end, secret, found);
}
