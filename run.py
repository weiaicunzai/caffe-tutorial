import os
import time
import random
from collections import Counter
import argparse

import GPUtil

class GPUStats:
    def __init__(self, num_gpus, sleep_time=50, exec_thresh=3, max_gpu_mem=0.01, max_gpu_util=0.01):
        self.sleep_time = sleep_time
        self.exec_thresh = exec_thresh
        self.max_gpu_mem = max_gpu_mem
        self.max_gpu_util = max_gpu_util
        self.num_gpus = num_gpus

        self.counter = Counter()

    def lookup(self):
        return GPUtil.getAvailable(order='first', limit=self.num_gpus, maxLoad=self.max_gpu_util, maxMemory=self.max_gpu_mem)

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

    def run(self, command, exec_gpu_id=None):
        while True:
            self.sleep()
            self.accumulate()

            for key, value in self.counter.items():
                if exec_gpu_id is not None:
                    if key != exec_gpu_id:
                        continue

                if value >= self.exec_thresh:
                    gpu_string = 'export CUDA_VISIBLE_DEVICES={}'.format(key)
                    print(gpu_string + '&&' + command)
                    os.system(gpu_string + '&&' + command)
                    return


if '__main__' == __name__:
    "current script only support single GPU training"
    parser = argparse.ArgumentParser()
    parser.add_argument('--exec_gpu_id',  type=int, default=None, help='the gpu id which we would like to execute our program')
    parser.add_argument('--command',  type=str, required=True, help='excute command')
    parser.add_argument('--sleep_time',  type=int, default=50)
    parser.add_argument('--exec_thresh',  type=int, default=3)
    parser.add_argument('--max_gpu_mem',  type=float, default=0.01, help='max gpu memory used')
    parser.add_argument('--max_gpu_util',  type=float, default=0.01, help='max gpu memory used')
    parser.add_argument('--num_gpus',  type=int, required=True, help='total number of GPUs on this server')
    ##
    args = parser.parse_args()
    print(args)
    gpu_stats = GPUStats(num_gpus=args.num_gpus, sleep_time=args.sleep_time, exec_thresh=args.exec_thresh, max_gpu_mem=args.max_gpu_mem, max_gpu_util=args.max_gpu_util)
    gpu_stats.run(args.command, args.exec_gpu_id)