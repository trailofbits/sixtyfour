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

struct entry_t {
  uint64_t start;
  uint64_t stop;
  uint64_t nops;
  uint64_t which;
};

struct kernel_info_t {
  struct entry_t *hostArgs;
  uint64_t total_size;
};

extern __global__ void findit(uint64_t secret, entry_t *args);

// taken from https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc/
typedef struct device_info_ {
  int blocks;
  int threads;
  int id;
  int clockrate;
  cudaStream_t stream;
  int slice_index;
  bool finished;
  entry_t *da; // device pointer
} device_info_t;


#define MAX_CUDA_DEVICES 256
extern device_info_t CUDA_DEVICES[MAX_CUDA_DEVICES];

extern int avilable_gpus;

extern "C" int gpu_get_workers(void);