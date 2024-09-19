import time
import argparse
import json
import os
import pickle
from transformers import AutoTokenizer
import importlib.util
from encode_types import encode_config


def load_user_module(module_path):
    module_path.replace(".py", "")
    module_name = module_path.split("/")[-1]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    user_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_module)
    return user_module


def save_binary_file(data, output_fn):
    """Save the selected data to .bin and .idx files as dict objects."""
    bin_fn = output_fn
    idx_fn = output_fn.replace(".bin", ".idx")
    offsets = []
    current_offset = 0
    with open(bin_fn, "wb") as f:
        for item in data:
            encoded_data = pickle.dumps(item)
            f.write(encoded_data)
            offsets.append(current_offset)
            current_offset += len(encoded_data)
    with open(idx_fn, "wb") as f:
        pickle.dump(offsets, f)


def initialize(args):
    args.save_path = os.path.join(args.output_path, args.corpus_name)
    os.makedirs(args.save_path, exist_ok=True)
    config_cn = os.path.join(args.save_path, "config.json")
    with open(config_cn, "w") as f:
        json.dump({"args": {**vars(args)}}, f, indent=4)
    
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.vocab_size <= 1 << 16 or args.save_dtype == "int32", "int16 is can't cover the tokenizer vocab size"    
    return tokenizer


def main(args):
    tokenizer = initialize(args)
    data_loader_module = load_user_module(args.data_loader_path)
    data_loader = data_loader_module.data_loader()
    encode_func = encode_config[args.encode_type]["encode_func"]
    data_func = encode_config[args.encode_type]["data_func"]
    total_data_cnt = 0

    start_time = time.time()
    file_idx = 0  # 输出文件编号
    total_token_processed = 0  # 总共处理了多少token
    cur_tokens = 0  # 当前处理了多少token
    cur_data = []
    for step, raw_data in enumerate(data_loader):
        item, token_cnt = encode_func(tokenizer, raw_data)
        cur_data.append(item)
        cur_tokens += token_cnt
        total_token_processed += token_cnt

        # 每处理100条数据，输出一次运行效率
        if step % 100 == 0:  
            elapsed = time.time() - start_time
            second = int(time.time() - start_time)
            print(f"\rprocessed {total_token_processed:.2e} tokens, run for {(second // 3600):02d}:{(second // 60 % 60):02d}:{(second % 60):02d}, {(total_token_processed / elapsed):.2e} tokens/s", end="")
        
        # 每tokens_per_file个token需要将tokens_list写入文件
        if cur_tokens >= args.tokens_per_file:  
            data = data_func(cur_data, tokenizer, args)
            fn = os.path.join(args.save_path, f"{args.corpus_name}_{file_idx}.bin")
            save_binary_file(data, fn)
            total_data_cnt += len(data)
            file_idx += 1
            cur_tokens = 0
            cur_data = []
    
    # 处理最后一部分数据
    if cur_tokens > 0:
        data = data_func(cur_data, tokenizer, args)
        fn = os.path.join(args.save_path, f"{args.corpus_name}_{file_idx}.bin")
        save_binary_file(data, fn)
        total_data_cnt += len(data)
        file_idx += 1
        cur_tokens = 0
        cur_data = []

    with open(os.path.join(args.save_path, "summary.json"), "w") as f:
        json.dump({
            "total_token_without_padding": total_token_processed,
            "total_data_cnt": total_data_cnt,
        }, f, indent=4)
    print(f"\nprocessed {total_token_processed} tokens.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--data_loader_path", type=str, required=True)
    parser.add_argument("--corpus_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--merge_data", action="store_true")
    parser.add_argument("--encode_type", type=str, required=True)
    parser.add_argument("--save_dtype", type=str, choices=["int16", "int32"], default="int32")
    parser.add_argument("--tokens_per_file", type=int, default=5*10**8)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
