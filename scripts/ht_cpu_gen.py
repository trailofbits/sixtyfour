#!/usr/bin/env python3

import sys
import multiprocessing as mp

m = mp.cpu_count() // 2
left = range(m)
right = range(m,m*2)
l = []
for x in zip(left,right):
  l.append(x[0])
  print(",".join(map(str, l)))
  l.append(x[1])
  print(",".join(map(str, l)))
