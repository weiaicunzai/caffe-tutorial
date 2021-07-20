```
COMMAND="nohup bash /home/baiyu/ViT-pytorch/run1.sh  &> /home/baiyu/ViT-pytorch/ecrc &"
  

python /home/baiyu/ViT-pytorch/run.py   --num_gpus  4   --command  "${COMMAND}"  --exec_gpu_id 3  --sleep_time 3
```
