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
#include "workpool.h"
#include "output.h"

#ifndef __FILE_NAME__
#define __FILE_NAME__ "WORKPOOL"
#endif

static volatile uint64_t POOL_START = 0;
static uint64_t POOL_END = UINT64_MAX;
static uint64_t CHUNK_SIZE = 0xFFFFFF;
static bool set_pool = false;

bool workpool_is_set(void) {
    return set_pool;
}
void workpool_set(uint64_t start, uint64_t end, unsigned workers) {

  POOL_START = start;
  POOL_END = end;
  // fairly arbitrary
  CHUNK_SIZE = (POOL_END - POOL_START) / workers / 16;
  log_output(__FILE_NAME__, "Using CHUNK_SIZE: [%" PRIx64 "]\n", CHUNK_SIZE);
  set_pool = true;
}

PoolStatus workpool_get_chunk(uint64_t *c_start, uint64_t *c_end) {

  // this thread's value of pool start
  uint64_t pool = POOL_START;
  // check if we're done
  uint64_t diff = POOL_END - pool;
  if (0 == diff) {
    log_output(__FILE_NAME__, "No more pool items\n");
    return PoolFinished;
  }

  // not done.. do we take a full chunk.... or a piece?
  uint64_t csize = (diff > CHUNK_SIZE) ? CHUNK_SIZE : diff;
  // figure out what the new value of 'start' should be assuming
  // no one else has claimed this chunk
  uint64_t new_pool = pool + csize;
  // see if someone beat us to claiming a piece of pool
  bool success = __sync_bool_compare_and_swap(&POOL_START, pool, new_pool);
  if (!success) {
    // another thread got here first
    return PoolTryAgain;
  }

  // victory.
  *c_start = pool;
  *c_end = new_pool;
  //log_output(__FILE_NAME__, "Fetched a pool item [%" PRIx64 "] - [%" PRIx64 "]\n",
  //  pool, new_pool);

  return PoolFetchedItem;
}