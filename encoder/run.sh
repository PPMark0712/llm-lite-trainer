declare -A corpus_loader_map=(
    ["alpaca"]="/data1/yyz/projects/llm-lite-trainer/encoder/user_data_loader/alpaca.py"
    ["math"]="/data1/yyz/projects/forget/encode/load_math.py"
)

for corpus_name in "${!corpus_loader_map[@]}"; do
    data_loader_path=${corpus_loader_map[$corpus_name]}
    
    python /data1/yyz/projects/llm-lite-trainer/encoder/main.py \
        --tokenizer_path /data2/dcy/downloads/model/meta-llama/Meta-Llama-3-8B \
        --data_loader_path $data_loader_path \
        --corpus_name $corpus_name \
        --encode_type qa \
        --output_path output/llama3 \
        --max_length 4096 \
        --merge_data \
        --save_dtype int32
done
