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

#pragma once
#include "common.h"

#ifdef __cplusplus
// NVCC can build as C++
extern "C"
{
#endif

typedef enum _PoolStatus {
    PoolFetchedItem,
    PoolTryAgain,
    PoolFinished
} PoolStatus;

bool workpool_is_set(void);
void workpool_set(uint64_t start, uint64_t end, unsigned workers);
PoolStatus workpool_get_chunk(uint64_t *c_start, uint64_t *c_end);

#ifdef __cplusplus
}
#endif