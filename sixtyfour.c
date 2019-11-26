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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdbool.h>

#ifndef __FILE_NAME__
#define __FILE_NAME__ "MAIN"
#endif

#include "timing.h"
#include "naive.h"
#include "multicore.h"
#include "sse.h"
#include "avx.h"
#include "avx512.h"
#include "gpu.h"
#include "gpu_multicore.h"

enum Methods {
  unknown,
  naive,
  avx2,
  multicore,
  sse4,
  avx512,
  gpu,
  gpu_multicore
};

typedef uint64_t (*method_func_t) (uint64_t, bool*, uint64_t, uint64_t);
typedef bool (*method_available_t) (void);
typedef struct method_info_ {
  enum Methods method;
  method_available_t available;
  const char *method_name;
  const char *description;
  method_func_t invoke;
} method_info_t;

method_info_t METHODS[] = {
  {naive,         naive_check,          "naive",          "Naive for loop         [1 64-bit quantity per iteration]", naive_method},
  {sse4,          sse_check,            "sse",            "SSE4.1 vectorized loop [2 64-bit quantities per iteration]", sse_method},
  {avx2,          avx2_check,           "avx2",           "AVX2 vectorized loop   [4 64-bit quantities per iteration]", avx2_method},
  {avx512,        avx512_check,         "avx512",         "AVX512 vectorized loop [8 64-bit quantities per iteration]", avx512_method},
  {multicore,     multicore_check,      "multicore",      "Best vectorized method in parallel on all cores", multicore_method},
  {gpu,           gpu_check,            "gpu",            "Run on all available NVIDIA GPUs", gpu_method},
  {gpu_multicore, gpu_multicore_check,  "gpu-multicore",  "Use all available GPUs and CPUs", gpu_multicore_method},
};

static void dump_methods() {
  for (unsigned i = 0; i < sizeof(METHODS) / sizeof(method_info_t); i++) {
    if (METHODS[i].available()) {
      printf("%-13s\n", METHODS[i].method_name);
    }
  }
}

static void usage(const char *pn) {
    printf("Usage: %s --methods\n", pn);
    printf("Usage: %s [-v] [--cpus 0,1,2,3 | --ncpu NUM] [--ngpu NUM] <method> <needle> [haystack start: default 0] [haystack end: default UINT64_MAX]\n", pn);
    printf("\n");
    printf("Try: %s -v naive 0xFF0000000 0x0 0xFFF000000\n", pn);
    printf("Available Methods:\n");
    for(unsigned i = 0; i < sizeof(METHODS)/sizeof(method_info_t); i++) {
      if(METHODS[i].available()) {
        printf("%-13s:\t %s\n", METHODS[i].method_name, METHODS[i].description);
      }
    }
}

static int parse_an_arg(const char *argv[], int idx, int argc) {
  static bool set_map = false;
  int args_consumed = 0;

  if(idx >= argc) {
    return 0;
  }

  if (0 == strcmp("-v", argv[idx])) {
    log_output = log_stdout;
    args_consumed += 1;
  }

  if (0 == strcmp("--methods", argv[idx])) {
    dump_methods();
    args_consumed += 1;
    exit(0);
  }

  if(0 == strcmp("--cpus", argv[idx])) {
    // can't use --cpus and --ncpu at once, even if this fails later
    set_map = true;
    if(idx + 1 >= argc) {
      usage(argv[0]);
      exit(1);
    }

    if(CPU_LIMIT > 0) {
      log_error(__FILE_NAME__, "Cannot use --cpus and --ncpu together");
      exit(1);
    }

    if(false == multicore_set_cpu_map(argv[idx+1])) {
      log_error(__FILE_NAME__, "Could not set CPU map!\n");
      exit(-1);
    } else {
      multicore_print_cpu_map();
    }
    args_consumed += 2;
  }

  if(0 == strcmp("--ncpu", argv[idx])) {
    if(set_map) {
      log_error(__FILE_NAME__, "Can't use --ncpu and --cpus at the same time\n");
      exit(1);
    }
    if(idx + 1 >= argc) {
      usage(argv[0]);
      exit(1);
    }
    int cpu_limit = strtol(argv[idx+1], NULL, 0);
    if(cpu_limit > 0) {
      log_output(__FILE_NAME__, "Setting CPU limit to %d\n", cpu_limit);
      multicore_init_cpus(cpu_limit);
    }
    args_consumed += 2;
  }

  if(0 == strcmp("--ngpu", argv[idx])) {
    if(idx + 1 >= argc) {
      usage(argv[0]);
      exit(1);
    }
    int gpu_limit = strtol(argv[idx+1], NULL, 0);
    if(gpu_limit > 0) {
      log_output(__FILE_NAME__, "Setting GPU limit to %d\n", gpu_limit);
      GPU_LIMIT = gpu_limit;
    }
    args_consumed += 2;
  }

  return args_consumed;
}

int main(int argc, const char *argv[]) {

  int arg_idx = 1;
  if(arg_idx >= argc) {
    fprintf(stderr, "Not enough arguments [%d].\n", arg_idx);
    usage(argv[0]);
    return -1;
  }

  // use all CPUs by default
  multicore_init_cpus(MAX_CPU);
  int consumed = 0;
  do {
    consumed = parse_an_arg(argv, arg_idx, argc);
    arg_idx += consumed;
  } while(consumed > 0);

  if(arg_idx >= argc) {
    fprintf(stderr, "Not enough arguments [%d].\n", arg_idx);
    usage(argv[0]);
    return 1;
  }

  const char *method_str = argv[arg_idx];
  arg_idx += 1;
  if(arg_idx >= argc) {
    fprintf(stderr, "Not enough arguments [%d].\n", arg_idx);
    usage(argv[0]);
    return 1;
  }

  uint64_t secret = strtoull(argv[arg_idx], NULL, 0);
  if(secret == 0 && 0 == strcmp("MAX", argv[arg_idx])) {
    secret = UINT64_MAX;
  }
  arg_idx += 1;

  if(arg_idx >= argc) {
    fprintf(stderr, "Not enough arguments [%d].\n", arg_idx);
    usage(argv[0]);
    return 1;
  }
  uint64_t hay_start = 0;
  uint64_t hay_end = UINT64_MAX;

  // init the output mutex
  initialize_output();
  hay_start = strtoull(argv[arg_idx], NULL, 0);
  arg_idx += 1;

  if(arg_idx >= argc) {
    fprintf(stderr, "Not enough arguments [%d].\n", arg_idx);
    usage(argv[0]);
    return 1;
  }

  // shortcut so people don't have to count the number of Fs
  if (0 == strcmp(argv[arg_idx], "MAX")) {
    hay_end = UINT64_MAX;
  } else {
    hay_end = strtoull(argv[arg_idx], NULL, 0);
  }
  arg_idx += 1;

  if(hay_end < hay_start) {
    log_error(__FILE_NAME__, "Error: Haystack end [%" PRIx64 "] must be before haystack start [%" PRIx64 "]\n",
        hay_end,
        hay_start);
  } 

  log_output(__FILE_NAME__, "Checking range from: [%016" PRIx64 "] - [%016" PRIx64 "]\n",
     hay_start,
     hay_end);

  method_info_t *meth = NULL;

  for(unsigned i = 0; i < sizeof(METHODS)/sizeof(method_info_t); i++) {
    if(0 == strcmp(method_str, METHODS[i].method_name)) {
      meth = &METHODS[i];
    }
  }

  if (NULL == meth) {
    log_error(__FILE_NAME__, "Method [%s] was not a known method\n",
              method_str);
    usage(argv[0]);
    return -1;
  }

  log_output(__FILE_NAME__,
             "Using method [%s] to look for: [0x%016" PRIx64 "]\n",
             meth->method_name, secret);

  uint64_t opscount = 0;
  bool found = false;

  gpu_get_device_info();

  elapsed_time_t perf_timer;
  start_timer(&perf_timer);
  opscount = meth->invoke(secret, &found, hay_start, hay_end);
  stop_timer(&perf_timer);
  

  if(found) {
    log_output(__FILE_NAME__, "Found secret!\n");
  } else {
    log_output(__FILE_NAME__, "Secret not found in search space\n");
  }
  show_elapsed_time(&perf_timer, opscount, meth->method_name);

  return 0;
}
