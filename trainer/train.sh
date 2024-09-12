deepspeed --include localhost:0,1,2,3 --master_port 12345 train.py \
    --model_path /data1/yyz/downloads/models/Qwen/Qwen2-7B \
    --data_path /data1/yyz/projects/forget/train/dataset/alpaca \
    --save_name "Qwen2-7B-alpaca" \
    --max_epochs 100 \
    --max_steps 10000 \
    --save_steps 2000 \