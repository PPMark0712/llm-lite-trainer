cd /data1/yyz/projects/llm-lite-trainer/trainer/

deepspeed --include localhost:0 --master_port 12345 train.py \
    --model_path /data1/yyz/downloads/models/Qwen/Qwen2-7B \
    --data_path /data1/yyz/projects/forget/train/dataset/alpaca \
    --save_name "Qwen2-7B-alpaca" \
    --max_epochs 3 \
    --save_epochs 2 \