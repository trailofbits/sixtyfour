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

#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>

typedef struct method_results_ {
  uint64_t ops_done;
  bool found;
} method_results_t;

typedef struct thread_args_ {
  pthread_t tid;
  uint64_t start;
  uint64_t stop;
  uint64_t secret;
  bool found;
  uint64_t nops;
  bool done;
  long ncpu;
} thread_args_t;