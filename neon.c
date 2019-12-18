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
  (void)start;
  (void)end;
  (void)secret;
  *found = false;
  return 0;
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
