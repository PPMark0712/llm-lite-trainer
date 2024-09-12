import heapq
import numpy as np


def encode_qa(tokenizer, conversation: dict):
    q_ids = tokenizer(conversation["q"])["input_ids"]
    a_ids = tokenizer(conversation["a"])["input_ids"]
    input_ids = q_ids + a_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(q_ids) + a_ids + [tokenizer.eos_token_id]
    token_cnt = len(input_ids)
    return (input_ids, labels), token_cnt


def data_func_qa(data, tokenizer, args):
    save_dtype = np.int16 if args.save_dtype == "int16" else np.int32
    if args.merge_data:
        merged_data = []
        heapq.heapify(merged_data)  # 初始化空堆
        
        # 用优先队列模拟best-fit法求解装箱问题，近似比约1.22
        for input_ids, labels in data:
            if len(input_ids) >= args.max_length:
                continue  # 丢弃过长数据
            # 新来的数据尝试和堆中最短的数据合并，若无法合并则新建一条数据
            if merged_data and len(merged_data[0][0]) + len(input_ids) <= args.max_length:
                curr_input_ids, curr_labels = heapq.heappop(merged_data)
                curr_input_ids += input_ids
                curr_labels += labels
                heapq.heappush(merged_data, (curr_input_ids, curr_labels))
            else:
                heapq.heappush(merged_data, (input_ids, labels))
        
        ret_data = []
        # padding & change dtype
        for input_ids, labels in merged_data:
            pad_len = args.max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            input_ids = np.array(input_ids, dtype=save_dtype)
            labels = np.array(labels, dtype=save_dtype)
            ret_data.append((input_ids, labels))
        
        return ret_data
    else:
        ret_data = []
        for input_ids, labels in data:
            if len(input_ids) > args.max_length:
                continue
            pad_len = args.max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            input_ids = np.array(input_ids, dtype=save_dtype)
            labels = np.array(labels, dtype=save_dtype)
            ret_data.append((input_ids, labels))
        
        return ret_data
