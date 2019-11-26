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

#include "output.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "gpu.h"
#include "gpu_multicore.h"

#ifndef __FILE_NAME__
#define __FILE_NAME__ "GPU_STUB"
#endif

int GPU_LIMIT = 0;

void gpu_get_device_info(void) {
  // do nothing
  return;
}

bool gpu_check(void) {
  return false;
}

bool gpu_multicore_check(void) {
  return false;
}

uint64_t gpu_method(uint64_t secret, bool *found, uint64_t h_start, uint64_t h_end) {
  (void)secret; // quash unused variable warnings
  (void)found;
  (void)h_start;
  (void)h_end;
  log_error(__FILE_NAME__, "GPU Method selected but the project was built without CUDA support\n");
  log_error(__FILE_NAME__, "Aborting\n");
  exit(-1);
}

uint64_t gpu_multicore_method(uint64_t secret, bool *found, uint64_t h_start, uint64_t h_end) {
  (void)secret; // quash unused variable warnings
  (void)found;
  (void)h_start;
  (void)h_end;
  log_error(__FILE_NAME__, "GPU_MULTICORE Method selected but the project was built without CUDA support\n");
  log_error(__FILE_NAME__, "Aborting\n");
  exit(-1);
}
