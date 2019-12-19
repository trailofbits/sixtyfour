# SixtyFour

How fast can we brute force a comparison against a 64-bit integer? 

Can we get it to a reasonable time with easily accessible computation? How much will it cost?

This code accompanies a Trail of Bits blog post, [64 bits ought to be enough for anybody!](https://blog.trailofbits.com/2019/11/27/64-bits-ought-to-be-enough-for-anybody/), that discusses those topics and more.

The `sixtyfour` tool measures how fast comparisons can happen using:

* A naive for loop. [see `naive.c`]
* SSE4.1 vectorization. [see `sse.c`]
* AVX2 vectorization. [see `avx.c`]
* AVX-512 vectorization. [see `avx512.c`]
* ARM NEON vectorization. [see `neon.c`]
* Multiple cores. [see `multicore.c`]
* GPUs using CUDA. [see `gpu.cu`]
* GPUs and CPUs working together. [see `gpu_multicore.cu`]

## Notes

This code is research / proof-of-concept quality and likely has multiple bugs. Please do not use it as a benchmark.

The methods implemented here are not guaranteed to be the fastest possible implementation. If you have a faster way, please submit a pull request.

The CUDA code in particular is "my first CUDA tutorial" level, which is exactly what it is based on. Improvements are welcome.

## Building

This code builds on Linux and MacOS with recent versions of Clang and GCC. All builds happen with CMake:
```sh
# Build withOUT GPU acceleration
mkdir build && cd build
cmake -DCUDA_BUILD=False ..
cmake --build .
```

On Linux, it can also be built with CUDA support to enable GPU acceleration. The build has only been tested with v10 of the CUDA SDK:
```sh
# Build with GPU acceleration
mkdir build && cd build
cmake -DCUDA_BUILD=True ..
cmake --build .
```

Sometimes CMake can't find `nvcc`; in that case, try manually specifying the path via: 
```
CUDACXX=/usr/local/cuda/bin/nvcc cmake -DCUDA_BUILD=True ..
```

Sixtyfour should build on ARMv864 (AArch64) and AMD64 (x86-64) platforms.

## Pull Requests Wanted

Some things I would love to add but do not have time for:

* Support for OpenCL.
* Fixes to the many bugs I am sure lurk in this code.

## Usage

Usage is as follows:

```
./sixtyfour [-v] [--cpus 0,1,2,3 | --ncpu NUM] [--ngpu NUM] <method> <needle> [haystack start: default 0] [haystack end: default UINT64_MAX]
```

The default output is timing data in JSON format. Try using `-v` for more verbose output and logging.

Things to try:
```
./sixtyfour --methods
./sixtyfour -v naive 0xFF0000000 0x0 0xFFF000000
./sixtyfour -v multicore 0xFF0000000 0x0 0xFFF000000
```

Not all methods are available on all platforms. By default `sixtyfour` will try to detect possible methods for your platform, and only show the ones it can run.

## Workload

The initial workload is split into chunks and distributed to workers as each one finishes. This is a bit slower than the naive approach of fixed-size chunks, but works best for heterogenous computing environments like GPUs and CPUs. See the implementation in `workpool.c` for more details.

# The Methods

This provides a brief overview of each method and an invocation output for each.

## Naive

On a 2018 MacBook Pro (Core i7 8559U, 2.7 Ghz):

```
$ ./sixtyfour -v naive 0xFF0000000 0x0 0xFFF00000
sixtyfour.c: Checking range from: [0000000000000000] - [00000000fff00000]
sixtyfour.c: Using method [naive] to look for: [0x0000000ff0000000]
sixtyfour.c: Secret not found in search space
timing.c: Method: naive
timing.c: Elapsed Time: 00:00:01.039
timing.c: Estimated ops per:
    millisecond:              4132742
         second:           4132741790
         minute:         247964507411
           hour:       14877870444658
            day:      357068890671800
timing.c: Time to search all 64-bits: 0141Y 196D 14H 13M 30S
{
  "finishwhen": "0141Y 196D 14H 13M 30S",
  "method": "naive",
  "etime": "00:00:01.039",
  "ms":  "4132741.790183",
  "sec": "4132741790.182868",
  "min": "247964507410.972089",
  "hour":"14877870444658.325313",
  "day": "357068890671799.807495"
}
```

## SSE

Throwing some SIMD into the mix to compare multiple 64-bit quantities at once. 

MacBook Pro (Core i7 8559U, 2.7 Ghz):
```
$ ./sixtyfour -v sse 0xFF0000000 0x0 0xFFF00000
sixtyfour.c: Checking range from: [0000000000000000] - [00000000fff00000]
sixtyfour.c: Using method [sse] to look for: [0x0000000ff0000000]
sixtyfour.c: Secret not found in search space
timing.c: Method: sse
timing.c: Elapsed Time: 00:00:00.667
timing.c: Estimated ops per:
    millisecond:              6437659
         second:           6437659250
         minute:         386259555022
           hour:       23175573301349
            day:      556213759232384
timing.c: Time to search all 64-bits: 0090Y 314D 20H 21M 08S
{
  "finishwhen": "0090Y 314D 20H 21M 08S",
  "method": "sse",
  "etime": "00:00:00.667",
  "ms":  "6437659.250375",
  "sec": "6437659250.374813",
  "min": "386259555022.488756",
  "hour":"23175573301349.325336",
  "day": "556213759232383.808075"
}
```

## AVX2

Using AVX2, it is possible to compare four 64-bit quantities at once.

MacBook Pro (Core i7 8559U, 2.7 Ghz):
```
$ ./sixtyfour -v avx2 0xFF0000000 0x0 0xFFF00000
sixtyfour.c: Checking range from: [0000000000000000] - [00000000fff00000]
sixtyfour.c: Using method [avx2] to look for: [0x0000000ff0000000]
sixtyfour.c: Secret not found in search space
timing.c: Method: avx2
timing.c: Elapsed Time: 00:00:00.317
timing.c: Estimated ops per:
    millisecond:             13545485
         second:          13545484921
         minute:         812729095268
           hour:       48763745716088
            day:     1170329897186120
timing.c: Time to search all 64-bits: 0043Y 067D 00H 06M 45S
{
  "finishwhen": "0043Y 067D 00H 06M 45S",
  "method": "avx2",
  "etime": "00:00:00.317",
  "ms":  "13545484.921136",
  "sec": "13545484921.135647",
  "min": "812729095268.138801",
  "hour":"48763745716088.328075",
  "day": "1170329897186119.873779"
}
```

## AVX-512

Intel's latest processors have AVX-512, allowing for 8 64-bit quantities to be compared at the same time.


DigitalOcean High-CPU Instance:
```
$ ./sixtyfour -v avx512 0xFF0000000 0x0 0xFFF00000
MAIN: Checking range from: [0000000000000000] - [00000000fff00000]
MAIN: Using method [avx512] to look for: [0x0000000ff0000000]
MAIN: Secret not found in search space
TIMING: Method: avx512
TIMING: Elapsed Time: 00:00:00.339
TIMING: Estimated ops per:
    millisecond:             12666427
         second:          12666426903
         minute:         759985614159
           hour:       45599136849558
            day:     1094379284389381
TIMING: Time to search all 64-bits: 0046Y 065D 21H 32M 51S
{
  "finishwhen": "0046Y 065D 21H 32M 51S",
  "method": "avx512",
  "etime": "00:00:00.339",
  "ms":  "12666426.902655",
  "sec": "12666426902.654867",
  "min": "759985614159.292035",
  "hour":"45599136849557.522121",
  "day": "1094379284389380.530884"
}
```

## ARM NEON

ARM's NEON only has 128-bit wide registers, but there are 32 of them. 

Result from an NVIDIA Jetson TX2:
```
$ ./sixtyfour -v neon 0xFF0000000 0x0 0xFFF00000
MAIN: Checking range from: [0000000000000000] - [00000000fff00000]
MAIN: Using method [neon] to look for: [0x0000000ff0000000]
MAIN: Secret not found in search space
TIMING: Method: neon
TIMING: Elapsed Time: 00:00:01.686
TIMING: Estimated ops per:
    millisecond:              2546808
         second:           2546808256
         minute:         152808495374
           hour:        9168509722420
            day:      220044233338078
TIMING: Time to search all 64-bits: 0229Y 246D 23H 45M 20S
{
  "finishwhen": "0229Y 246D 23H 45M 20S",
  "method": "neon",
  "etime": "00:00:01.686",
  "ms":  "2546808.256228",
  "sec": "2546808256.227758",
  "min": "152808495373.665480",
  "hour":"9168509722419.928826",
  "day": "220044233338078.291815"
}
```

## Multicore
Uses the best available single core method parallelized across multiple cores. The `-ncpu` and `-cpus` options select how many and which CPUs to use. By default all available CPUs are utilized.

On a 32 vCPU DigitalOcean High-CPU Instance:

```
$ ./sixtyfour -v multicore MAX 0x0 0xFFF0000000
MAIN: Checking range from: [0000000000000000] - [000000fff0000000]
MAIN: Using method [multicore] to look for: [0xffffffffffffffff]
MULTICORE: Found [32] enabled processors
WORKPOOL: Using CHUNK_SIZE: [7ff80000]
MULTICORE: Starting threads
MULTICORE: Thread will use CPU: 0
MULTICORE: .
MULTICORE: Thread will use CPU: 1
MULTICORE: .
MULTICORE: Thread will use CPU: 2
MULTICORE: .
MULTICORE: Thread will use CPU: 3
MULTICORE: .
MULTICORE: Thread will use CPU: 4
MULTICORE: .
MULTICORE: Thread will use CPU: 5
MULTICORE: .
MULTICORE: Thread will use CPU: 6
MULTICORE: .
MULTICORE: Thread will use CPU: 7
MULTICORE: .
MULTICORE: Thread will use CPU: 8
MULTICORE: .
MULTICORE: Thread will use CPU: 9
MULTICORE: .
MULTICORE: Thread will use CPU: 10
MULTICORE: .
MULTICORE: Thread will use CPU: 11
MULTICORE: .
MULTICORE: Thread will use CPU: 12
MULTICORE: .
MULTICORE: Thread will use CPU: 13
MULTICORE: .
MULTICORE: Thread will use CPU: 14
MULTICORE: .
MULTICORE: Thread will use CPU: 15
MULTICORE: .
MULTICORE: Thread will use CPU: 16
MULTICORE: .
MULTICORE: Thread will use CPU: 17
MULTICORE: .
MULTICORE: Thread will use CPU: 18
MULTICORE: .
MULTICORE: Thread will use CPU: 19
MULTICORE: .
MULTICORE: Thread will use CPU: 20
MULTICORE: .
MULTICORE: Thread will use CPU: 21
MULTICORE: .
MULTICORE: Thread will use CPU: 22
MULTICORE: .
MULTICORE: Thread will use CPU: 23
MULTICORE: .
MULTICORE: Thread will use CPU: 24
MULTICORE: .
MULTICORE: Thread will use CPU: 25
MULTICORE: .
MULTICORE: Thread will use CPU: 26
MULTICORE: .
MULTICORE: Thread will use CPU: 27
MULTICORE: .
MULTICORE: Thread will use CPU: 28
MULTICORE: .
MULTICORE: Thread will use CPU: 29
MULTICORE: .
MULTICORE: Thread will use CPU: 30
MULTICORE: .
MULTICORE: Thread will use CPU: 31
MULTICORE: .
MULTICORE:
MULTICORE: Waiting for worker threads
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af2cea700] done [40792227840] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af6cf2700] done [36498309120] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af8cf6700] done [25763512320] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af44ed700] done [40792227840] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af24e9700] done [30057431040] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4afb4fb700] done [27910471680] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af0ce6700] done [30057431040] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4afbcfc700] done [34351349760] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af74f3700] done [19322634240] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af04e5700] done [36498309120] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4aef4e3700] done [36498309120] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4aefce4700] done [36498309120] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af34eb700] done [38645268480] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4aee4e1700] done [38645268480] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4aed4df700] done [38645268480] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4aedce0700] done [38645268480] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af3cec700] done [38645268480] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4aeece2700] done [38645268480] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af7cf4700] done [47233105920] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af84f5700] done [23616552960] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af64f1700] done [30057431040] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4afc4fd700] done [38645268480] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af1ce8700] done [36498309120] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af9cf8700] done [25763512320] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af4cee700] done [32204390400] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4afccfe700] done [34351349760] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af5cf0700] done [45086146560] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4afa4f9700] done [27910471680] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af94f7700] done [42939187200] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4afacfa700] done [27910471680] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af14e7700] done [36498309120] ops using [AVX512]
WORKPOOL: No more pool items
MULTICORE: Thread [7f4af54ef700] done [23616552960] ops using [AVX512]
MULTICORE: Finished. Stopping threads...
MULTICORE: Waiting for threads to finish last work items
MAIN: Secret not found in search space
TIMING: Method: multicore
TIMING: Elapsed Time: 00:00:04.939
TIMING: Estimated ops per:
    millisecond:            222563918
         second:         222563918267
         minute:       13353835096011
           hour:      801230105760680
            day:    19229522538256327
TIMING: Time to search all 64-bits: 0002Y 229D 07H 01M 45S
{
  "finishwhen": "0002Y 229D 07H 01M 45S",
  "method": "multicore",
  "etime": "00:00:04.939",
  "ms":  "222563918.266856",
  "sec": "222563918266.855639",
  "min": "13353835096011.338327",
  "hour":"801230105760680.299622",
  "day": "19229522538256327.191406"
}
```


## GPU 

These days our machines also have a GPU, why waste it?  A GeForce GT 1030 (cost: ~$85 USD) about as fast as 40 Sandy Bridge Xeons. Fortunately, this problem is embarassingly parallel and GPU friendly.

GPU: GeForce GT 1030 (384 Cuda Cores, 1.47Ghz):
```
$ ./sixtyfour -v gpu MAX 0x0 0xFFF000000
MAIN: Checking range from: [0000000000000000] - [0000000fff000000]
MAIN: Using method [gpu] to look for: [0xffffffffffffffff]
GPU_COMMON: Found 1 devices
GPU_COMMON: GPU[0]: Device Number: 0
GPU_COMMON: GPU[0]:   Device name: GeForce GT 1030
GPU_COMMON: GPU[0]:   Clock Rate (KHz): 1468000
GPU_COMMON: GPU[0]:   Blocks: 6
GPU_COMMON: GPU[0]:   Threads: 1024
WORKPOOL: Using CHUNK_SIZE: [aaa00]
GPU: GPU: Using 6 blocks of 1024 threads over 1 GPUs
GPU: GPU[0] Filling work slices [000000 - 006143]
GPU: GPU[0]: Operations complete

[... omitted for brevity ...]

GPU: GPU[0]: Saving results for slices [000000] - [006143]
GPU: GPU[0] Filling work slices [000000 - 006143]
WORKPOOL: No more pool items
GPU: GPU: Performed: [68702699520] operations
MAIN: Secret not found in search space
TIMING: Method: gpu
TIMING: Elapsed Time: 00:00:00.680
TIMING: Estimated ops per:
    millisecond:            101033382
         second:         101033381647
         minute:        6062002898824
           hour:      363720173929412
            day:     8729284174305882
TIMING: Time to search all 64-bits: 0005Y 288D 04H 51M 26S
{
  "finishwhen": "0005Y 288D 04H 51M 26S",
  "method": "gpu",
  "etime": "00:00:00.680",
  "ms":  "101033381.647059",
  "sec": "101033381647.058824",
  "min": "6062002898823.529412",
  "hour":"363720173929411.764709",
  "day": "8729284174305882.353027"
}
```

## GPUs and CPUs, together

Although GPUs are much better at this problem (an order of magnitude better), it is possible to try to use some of the spare CPU capacity. Usually though this just makes things slower than using only GPUs, but there are situations where it could be useful.

On a Core i3 530 and a GeForce GT 1030:

```
$ ./sixtyfour -v gpu-multicore MAX 0x0 0xFFF000000
MAIN: Checking range from: [0000000000000000] - [0000000fff000000]
MAIN: Using method [gpu-multicore] to look for: [0xffffffffffffffff]
GPU_COMMON: Found 1 devices
GPU_COMMON: GPU[0]: Device Number: 0
GPU_COMMON: GPU[0]:   Device name: GeForce GT 1030
GPU_COMMON: GPU[0]:   Clock Rate (KHz): 1468000
GPU_COMMON: GPU[0]:   Blocks: 6
GPU_COMMON: GPU[0]:   Threads: 1024
GPU_MULTICORE: Total workers = [6148] [CPU: 4][GPU: 6144]
WORKPOOL: Using CHUNK_SIZE: [aa839]
GPU_MULTICORE: Creating worker threads
GPU_MULTICORE: .
GPU: GPU: Using 6 blocks of 1024 threads over 1 GPUs
MULTICORE: Found [4] enabled processors
MULTICORE: Starting threads
MULTICORE: Thread will use CPU: 0
MULTICORE: .
GPU_MULTICORE: .
GPU_MULTICORE: Waiting on a method to finish...
MULTICORE: Thread will use CPU: 1
MULTICORE: .
MULTICORE: Thread will use CPU: 2
MULTICORE: .
MULTICORE: Thread will use CPU: 3
MULTICORE: .
MULTICORE:
MULTICORE: Waiting for worker threads
GPU: GPU[0] Filling work slices [000000 - 006143]
[ ... omitted for brevity ... ]
GPU: GPU[0]: Operations complete
GPU: GPU[0]: Saving results for slices [000000] - [006143]
GPU: GPU[0] Filling work slices [000000 - 006143]
WORKPOOL: No more pool items
WORKPOOL: No more pool items
MULTICORE: Thread [7faaedfff700] done [688638176] ops using [SSE4.1]
WORKPOOL: No more pool items
MULTICORE: Thread [7faaed7fe700] done [685844512] ops using [SSE4.1]
WORKPOOL: No more pool items
MULTICORE: Thread [7faaf4ffd700] done [696320752] ops using [SSE4.1]
WORKPOOL: No more pool items
MULTICORE: Thread [7faaf57fe700] done [666288864] ops using [SSE4.1]
MULTICORE: Finished. Stopping threads...
MULTICORE: Waiting for threads to finish last work items
GPU_MULTICORE: A thread finished!
GPU: GPU: Performed: [65965575799] operations
GPU_MULTICORE: A thread finished!
GPU_MULTICORE: All threads done!
GPU_MULTICORE: There were [0000068702668103] operations performed
GPU_MULTICORE: 	[GPU] did [0000065965575799] Operations, or [96.02] percent
GPU_MULTICORE: 	[CPU] did [0000002737092304] Operations, or [3.98] percent
MAIN: Secret not found in search space
TIMING: Method: gpu-multicore
TIMING: Elapsed Time: 00:00:00.680
TIMING: Estimated ops per:
    millisecond:            101033335
         second:         101033335446
         minute:        6062000126735
           hour:      363720007604118
            day:     8729280182498824
TIMING: Time to search all 64-bits: 0005Y 288D 04H 52M 49S
{
  "finishwhen": "0005Y 288D 04H 52M 49S",
  "method": "gpu-multicore",
  "etime": "00:00:00.680",
  "ms":  "101033335.445588",
  "sec": "101033335445.588235",
  "min": "6062000126735.294117",
  "hour":"363720007604117.647034",
  "day": "8729280182498823.528809"
}
```
