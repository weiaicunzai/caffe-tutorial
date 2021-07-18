import os
import time
import random
from collections import Counter
import argparse

import GPUtil

#def lookup(max_gpu_util=0.01, max_gpu_mem=0.01):
#    return GPUtil.getAvailable(order='first', limit=2, maxLoad=1, maxMemory=1)
#
#def accumulate(records, cur_gpus):
#
#    for gpu in range(len(records)):
#        if gpu not in cur_gpus:
#            records[gpu] = 0
#        else:
#            records[gpu] += 1
#
#    return records
#
#def sleep_time():
#    return random.gauss(50, 10)


#def accumulate(counter, gpu_stats):
    #if len(gpu_stats) == 0:
        #counter.
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
        print(gpu_stats)
        #print(gpu_stats, type(gpu_stats[0]))
        #import sys; sys.exit()
        if len(gpu_stats) == 0:
            self.counter.clear()
        else:
            self.counter.update(gpu_stats)

        print(self.counter)

    def sleep(self):
        sleep_time = random.gauss(self.sleep_time, self.sleep_time / 5)
        time.sleep(sleep_time)

    def run(self, command, exec_gpu_id=None):
        while True:
            #sleep_time = self.sleep_time()
            #time.sleep(sleep_time)
            self.sleep()
            #print('fff')
            self.accumulate()

            for key, value in self.counter.items():
                #if exec_gpu_id == key:
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

#command = 'cd /data/by/pytorch-cifar100 &&' + \
          #'export CUDA_VISIBLE_DEVICES=1 &&' + \
          #'for i in 1 2 3 4\n' + \
          #'do\n' + \
          #'echo ${i}\n' + \
          #'python -u train.py -net stochasticdepth50 -gpu -b 256 -lr 0.1  -warm 5\n' + \
          #'done'

#os.system(command)
#records = [0, 0, 0, 0]
#records = [0, 0]
#while True:
#    t = sleep_time()
#
#    # approximate 600s to lookup gpu
#    time.sleep(1)
#
#    gpus = lookup()
#    records = accumulate(records, gpus)
#    os.system('date')
#    print(gpus)
#    print(records)
#
#    if records[0] > 3:
#        print('gpu0 is avilable')
#        #os.system('cd /data/by/pytorch-cifar100 && export CUDA_VISIBLE_DEVICES=1 && nohup sh run.sh &> nohup.out &')
#        os.system()
#        break
#
#    if records[1] > 3:
#        print('gpu0 is avilable')
#        os.system()
#        break
    #if records[2] > 3:
    #    print('gpu2 is avilable')
    #    os.system('cd /data/by/pytorch-cifar100 && nohup sh run.sh &> nohup.out &')
    #    break

#print(type(lookup()[0]))

#os.system('export CUDA_VISIBLE_DEVICES=0&&echo $CUDA_VISIBLE_DEVICES')
#os.system('echo $CUDA_VISIBLE_DEVICES')