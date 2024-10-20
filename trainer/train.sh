cd /data1/yyz/projects/llm-lite-trainer/trainer/

model_path=path/to/your/model
data_path=path/to/your/data
output_path=path/to/save/checkpoints
save_name=your_model_name  # for example: "Qwen2-7B-alpaca", "llama3-8B-Instruct-MetaMathQA"


deepspeed --include localhost:0,1,2,3 --master_port 12345 train.py \
    --model_path $model_path \
    --data_path $data_path \
    --output_path $output_path \
    --shuffle_data \
    --save_name $save_name \
    --max_epochs 1 \
    --save_steps 2000 \

