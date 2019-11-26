#!/bin/bash

# Automatically do several measurements of how fast this platform does comparisons

PROG=$(pwd)/build/sixtyfour
SECRET="MAX"
START=0

NCPU=$(python3 -c 'import multiprocessing as mp; print(mp.cpu_count())')
NGPU=0
if [ "$(which nvidia-smi)" != "" ]
then
  NGPU=$(nvidia-smi --query-gpu=count --format=noheader,csv | head -n1)
fi

BENCHDIR=measure_$(date +%s)


single_cpu_checks() {
  local my_rounds=${1}
  local my_limit=${2}

  for m in naive sse avx2 avx512
  do
    for round in $(seq ${my_rounds} -1 1)
    do
      echo "Trying ${PROG} $m ${SECRET} ${START} ${my_limit} > ${BENCHDIR}/${m}_${round}_cpu1_gpu0.out"
      local start_time=${SECONDS}
      ${PROG} $m ${SECRET} ${START} ${my_limit} > ${BENCHDIR}/${m}_${round}_cpu1_gpu0.out
      local end_time=${SECONDS}
      echo "Took $(( ${end_time} - ${start_time} )) seconds"

    done
  done
}

gpu_checks() {
  local my_rounds=${1}
  local my_limit=${2}

  for m in gpu
  do
    for i in $(seq ${NGPU} -1 1)
    do
      for round in $(seq ${my_rounds} -1 1)
      do
        echo "Trying ${PROG} --ngpu $i  $m ${SECRET} ${START} ${my_limit} > ${BENCHDIR}/${m}_${round}_cpu0_gpu${i}.out"
        local start_time=${SECONDS}
        ${PROG} --ngpu $i $m ${SECRET} ${START} ${my_limit} > ${BENCHDIR}/${m}_${round}_cpu0_gpu${i}.out
        local end_time=${SECONDS}
        echo "Took $(( ${end_time} - ${start_time} )) seconds"
      done
    done
  done

  for m in "gpu-multicore"
  do
    for round in $(seq ${my_rounds} -1 1)
    do
      echo "Trying ${PROG} $m ${SECRET} ${START} ${my_limit} > ${BENCHDIR}/${m}_${round}_cpu${NCPU}_gpu${NGPU}.out"
      local start_time=${SECONDS}
      ${PROG} $m ${SECRET} ${START} ${my_limit} > ${BENCHDIR}/${m}_${round}_cpu${NCPU}_gpu${NGPU}.out
      local end_time=${SECONDS}
      echo "Took $(( ${end_time} - ${start_time} )) seconds"
    done
  done
}

multicore_checks() {
  local my_rounds=${1}
  local my_limit=${2}

  for m in multicore
  do
    for i in $(seq ${NCPU} -1 1)
    do
      for round in $(seq ${my_rounds} -1 1)
      do
        echo "Trying ${PROG} --ncpu $i $m ${SECRET} ${START} ${my_limit} > ${BENCHDIR}/${m}_${round}_cpu${i}_gpu0.out"
        local start_time=${SECONDS}
        ${PROG} --ncpu $i $m  ${SECRET} ${START} ${my_limit} > ${BENCHDIR}/${m}_${round}_cpu${i}_gpu0.out
        local end_time=${SECONDS}
        echo "Took $(( ${end_time} - ${start_time} )) seconds"
      done
    done
  done
}

multicore_ht_checks() {
  local my_rounds=${1}
  local my_limit=${2}

  for m in multicore
  do
    for i in $($(pwd)/scripts/ht_cpu_gen.py | sort -r)
    do
      for round in $(seq ${my_rounds} -1 1)
      do
        echo "Trying ${PROG} --cpus $i $m ${SECRET} ${START} ${my_limit} > ${BENCHDIR}/${m}_${round}_cpumask${i}_gpu0.out"
        local start_time=${SECONDS}
        ${PROG} --cpus $i $m ${SECRET} ${START} ${my_limit} > ${BENCHDIR}/${m}_${round}_cpumask${i}_gpu0.out
        local end_time=${SECONDS}
        echo "Took $(( ${end_time} - ${start_time} )) seconds"
      done
    done
  done
}

main() {
  # These are different to account for
  # how much faster each version works.
  # A longer "work time" helps make 
  # a constant "setup time" negligible in results
  local single_limit=0xFFF0000000
  local  multi_limit=0xFFF00000000
  local    gpu_limit=0xFFF000000000

  mkdir -p ${BENCHDIR}
  
  if [ "$(uname -s)" = "Darwin" ]
  then
    sysctl machdep.cpu > ${BENCHDIR}/cpuinfo
  else
    cat /proc/cpuinfo >  ${BENCHDIR}/cpuinfo
    cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list >> ${BENCHDIR}/cpuinfo
  fi

  gpu_checks 10 ${gpu_limit}
  multicore_ht_checks 10 ${multi_limit}
  multicore_checks 10 ${multi_limit}
  single_cpu_checks 10 ${single_limit}
}

main

