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

#include "naive.h"

#ifndef __FILE_NAME__
#define __FILE_NAME__ "NAIVE"
#endif

bool naive_check(void) {
  return true;
}
// a simple comparison
static bool eval(uint64_t input, uint64_t secret) {
  return secret == input;
}

uint64_t naive_do_range(uint64_t start,
                        uint64_t end,
                        uint64_t secret,
                        bool *found)
{
  return naive_method(secret, found, start, end);
}

uint64_t naive_method(uint64_t secret, bool *found, uint64_t h_start, uint64_t h_end) {
  *found = false;

  const uint64_t start = h_start;
  const uint64_t end = h_end;

  uint64_t i = 0;
  for(i = start; i < end; i++) {
    if (true == eval(i, secret)) {
      *found = true;
      break;
    }
  }

  return i-start;
}
