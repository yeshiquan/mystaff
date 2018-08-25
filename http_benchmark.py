#!/usr/bin/env python
# encoding: utf-8

import json
import threading
import urllib2
import time
import argparse
from Queue import Queue
import numpy as np

cost_queue = Queue(maxsize=0)

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--concurrent", help="cocurrent")
parser.add_argument("-n", "--number", help="number")
parser.add_argument("-u", "--url", help="url")

args = parser.parse_args()

C = int(args.concurrent) if args.concurrent else 1
N = int(args.number) if args.number else 1
f = open(args.url, "r")
conf = json.loads(f.read().strip())
url = conf["url"]

start = time.time()

def fetch_url(url, num):
    for i in xrange(0, N):
        s = time.time()
        urlHandler = urllib2.urlopen(url)
        data = urlHandler.read()

        try:
            result = json.loads(data)
            if result['ret'] != 0:
                print data
                break
        except:
            print data
            break
            
        cost = time.time() - s
        cost_queue.put(cost*1000)
    #print "thread %d finished" % (num)

threads = [threading.Thread(target=fetch_url, args=(url,i)) for i in xrange(0,C)]
for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

seconds = time.time() - start

print "Elapsed Time: %.1f ms" % (seconds*1000)
print "QPS: %.1f" % (C*N/seconds)
print "Time per request: %.1f ms" % (seconds*1000/(C*N))

costs = []
while True:
    try:
        cost = cost_queue.get(False)
        costs.append(cost)
    except:
        break
if len(costs) > 0:
    a=np.array((costs))
    np.median(a)
    print "平均: %.1f ms" % (np.mean(a))
    print "90分位: %.1f ms" % (np.percentile(a, 90)) 
    print "95分位: %.1f ms" % (np.percentile(a, 95))
    print "最慢: %.1f ms" % (np.max(a))
