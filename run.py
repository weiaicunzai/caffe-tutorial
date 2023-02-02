import os
import time
import random
from collections import Counter
import argparse

import GPUtil

class GPUStats:
    def __init__(self, gpus_needed, sleep_time=50, exec_thresh=3, max_gpu_mem_avail=0.01, max_gpu_util=0.01):
        self.sleep_time = sleep_time
        self.exec_thresh = exec_thresh
        self.max_gpu_mem = max_gpu_mem_avail
        self.max_gpu_util = max_gpu_util
        self.gpus_needed = gpus_needed
        self.total_gpus = len(GPUtil.getGPUs())

        self.counter = Counter()

    def lookup(self):
        return GPUtil.getAvailable(order='memory', limit=self.total_gpus, maxMemory=self.max_gpu_mem)

    def accumulate(self):
        # [int, int, ...]
        gpu_stats = self.lookup()
        if len(gpu_stats) == 0:
            self.counter.clear()
        else:
            self.counter.update(gpu_stats)

    def sleep(self):
        sleep_time = random.gauss(self.sleep_time, self.sleep_time / 5)
        time.sleep(sleep_time)

    def if_available(self, return_gpu_ids=False):
        avail = []
        for gpu_id, count in self.counter.items():
            if count >= self.exec_thresh:
                avail.append(gpu_id)

        if return_gpu_ids:
            return len(avail) >= self.gpus_needed, avail
        return len(avail) >= self.gpus_needed

    def run(self, command):
        while True:
            self.sleep()
            self.accumulate()
            avai, gpu_ids = self.if_available(return_gpu_ids=True)

            if avai:
                gpu_ids = [str(gpu_id) for gpu_id in gpu_ids]
                gpu_string = 'export CUDA_VISIBLE_DEVICES={}'.format(','.join(gpu_ids))
                print(gpu_string + ' && ' + command)
                os.system(gpu_string + '&&' + command)
                return

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--command',  type=str, required=True, help='excute command')
    parser.add_argument('--sleep_time',  type=int, default=50)
    parser.add_argument('--exec_thresh',  type=int, default=3)
    parser.add_argument('--max_gpu_mem',  type=float, default=0.01, help='max gpu memory available')
    parser.add_argument('--max_gpu_util',  type=float, default=0.01, help='max gpu memory available')
    parser.add_argument('--num_gpus',  type=int, required=True, help='number of GPUs will use for this task')
    ##
    args = parser.parse_args()
    print(args)
    gpu_stats = GPUStats(gpus_needed=args.num_gpus, sleep_time=args.sleep_time, exec_thresh=args.exec_thresh, max_gpu_mem_avail=args.max_gpu_mem, max_gpu_util=args.max_gpu_util)
    gpu_stats.run(args.command)