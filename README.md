```
COMMAND="nohup bash /home/baiyu/ViT-pytorch/run1.sh  &> /home/baiyu/ViT-pytorch/ecrc &"


python -u run.py  --command  "${COMMAND}"  --num_gpus 2  --sleep_time 30 --max_gpu_mem 0.01

```

``--num_gpus: number of gpu used by the python program``


``--max_gpu_mem: memory already used in GPUs``


``--sleep_time: query interval``