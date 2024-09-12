python main.py \
    --tokenizer_path "/data1/yyz/downloads/models/Qwen/Qwen2-7B" \
    --data_loader_path /data1/yyz/projects/llm-lite-trainer/encoder/user_data_loader/alpaca.py \
    --corpus_name alpaca \
    --encode_type qa \
    --output_path output/debug_alpaca \
    --max_length 4096 \
    --merge_data \
    --save_dtype int32 \
