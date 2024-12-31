declare -A corpus_loader_map=(
    ["fincorpus_hard_sample_pos_ent"]="/home/yyz/projects/train_exp/encode/data_loaders/fincorpus_hard_sample_pos_ent.py pretrain"
    # ["fincorpus_mix_sample_mix_ent"]="/home/yyz/projects/train_exp/encode/data_loaders/fincorpus_mix_sample_mix_ent.py pretrain"
    # ["fincorpus_hard_sample_mix_ent"]="/home/yyz/projects/train_exp/encode/data_loaders/fincorpus_hard_sample_mix_ent.py pretrain"
)

declare -A tokenizer_path_map=(
    ["Qwen2.5"]="/mnt/remote-data/downloads/models/Qwen/Qwen2.5-7B-Instruct"
)

# Loop through the corpus_loader_map
for corpus_name in "${!corpus_loader_map[@]}"; do
    # Extract data_loader_path and encode_type by splitting the value
    data_loader_info=(${corpus_loader_map[$corpus_name]})
    data_loader_path=${data_loader_info[0]}
    encode_type=${data_loader_info[1]}

    for model_name in "${!tokenizer_path_map[@]}"; do
        tokenizer_path=${tokenizer_path_map[$model_name]}

        python encode_multiprocess.py \
            --tokenizer_path $tokenizer_path \
            --data_loader_path $data_loader_path \
            --corpus_name $corpus_name \
            --encode_type $encode_type \
            --max_length 4096 \
            --num_subprocesses 16 \
            --merge_data \
            --output_path output \
            --save_dtype int32 \
            --tokens_per_file 1000000000
    done
done
