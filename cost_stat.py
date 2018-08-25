#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

f = open(sys.argv[1], "r")
lines = f.read().strip().split("\n")
re = []
for line in lines:
    re.append(float(line))
a=np.array((re))
np.median(a)

print "平均: " + str(np.mean(a)) + "ms"
print "90分位: " + str(np.percentile(a, 90)) + "ms"
print "95分位: " + str(np.percentile(a, 95)) + "ms"
print "最慢: " + str(np.max(a)) + "ms"
