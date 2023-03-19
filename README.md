```
# first line of your code:

from gpustats import GPUStats
gpu_stats = GPUStats(gpus_needed=1, sleep_time=30, exec_thresh=3, max_gpu_mem_avail=0.01, max_gpu_util=0.01)
gpu_stats.run()

```

``gpus_needed: num of gpu needed to execute the program``


``sleep_time: sleep time for each query interval``


``exec_thresh: how many times of successful query needed before execute programs``

``max_gpu_mem_avail: max percentage of gpu mem already used``

``max_gpu_mem_util: max percentage of gpu mem util, similar to max_gpu_mem_avail``