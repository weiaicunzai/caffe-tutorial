import os
import time
import random
from collections import Counter
import argparse
import datetime

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

    def run(self):
        while True:
            self.sleep()
            self.accumulate()
            now = datetime.datetime.now()
            avai, gpu_ids = self.if_available(return_gpu_ids=True)
            print(now, avai, gpu_ids)

            if avai:
                gpu_ids = [str(gpu_id) for gpu_id in gpu_ids]
                os.environ('CUDA_VISIBLE_DEVICES') = ','.join(gpu_ids)
                return


#from gpustats import GPUStats
#gpu_stats = GPUStats(gpus_needed=1, sleep_time=30, exec_thresh=3, max_gpu_mem_avail=0.01, max_gpu_util=0.01)
#gpu_stats.run()