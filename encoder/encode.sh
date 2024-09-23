declare -A corpus_loader_map=(
    # ["alpaca"]="/data1/yyz/projects/llm-lite-trainer/encoder/user_data_loader/alpaca.py qa"
    # ["math"]="/data1/yyz/projects/forget/encode/load_math.py qa"
    # ["finexam"]="/data1/yyz/projects/forget/encode/load_finexam.py pretrain"
    ["pile"]="/data1/yyz/projects/forget/encode/load_pile.py pretrain"
)

declare -A tokenizer_path_map=(
    # ["llama3"]="/data2/dcy/downloads/model/meta-llama/Meta-Llama-3-8B"
    ["llama2"]="/data1/yyz/downloads/models/NousResearch/Llama-2-7b-hf"
    ["Qwen2"]="/data1/yyz/downloads/models/Qwen/Qwen2-7B"
)

# Loop through the corpus_loader_map
for corpus_name in "${!corpus_loader_map[@]}"; do
    # Extract data_loader_path and encode_type by splitting the value
    data_loader_info=(${corpus_loader_map[$corpus_name]})
    data_loader_path=${data_loader_info[0]}
    encode_type=${data_loader_info[1]}

    for model_name in "${!tokenizer_path_map[@]}"; do
        tokenizer_path=${tokenizer_path_map[$model_name]}

        python /data/yingyizhou/projects/llm-lite-trainer/encoder/main.py \
            --tokenizer_path $tokenizer_path \
            --data_loader_path $data_loader_path \
            --corpus_name $corpus_name \
            --encode_type $encode_type \
            --output_path /data/yingyizhou/projects/huawei-fin/data/encoded/$model_name \
            --max_length 4096 \
            --merge_data \
            --save_dtype int32 \
            --tokens_per_file 1000000000
    done
done
