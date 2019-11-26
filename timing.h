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

#include <time.h>
#include <stdint.h>

#ifdef __cplusplus
// NVCC can build as C++
extern "C"
{
#endif
// timer control functions
typedef struct elapsed_time_ {
    struct timespec start_time;
    struct timespec stop_time;
} elapsed_time_t;

void start_timer(elapsed_time_t *timer);
void stop_timer(elapsed_time_t *timer);
void show_elapsed_time(elapsed_time_t *timer, uint64_t nops, const char *mname);
uint64_t get_ops_per_ms(elapsed_time_t *timer, uint64_t nops);

#ifdef __cplusplus
// NVCC can build as C++
};
#endif