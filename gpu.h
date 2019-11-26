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
struct kernel_info_t;
uint64_t gpu_method(uint64_t secret, bool *found, uint64_t h_start, uint64_t h_end);
bool gpu_launch_kernel(struct kernel_info_t *ki, uint64_t secret, uint64_t h_start, uint64_t h_end);
void gpu_wait_for_streams(const struct kernel_info_t *ki, uint64_t secret, method_results_t *results);
bool gpu_check(void);

extern int GPU_LIMIT;
void gpu_get_device_info(void);

#ifdef __cplusplus
}
#endif