import numpy as np


def encode_qa(tokenizer, conversation:dict):
    q_ids = tokenizer(conversation["q"])["input_ids"]
    a_ids = tokenizer(conversation["a"])["input_ids"]
    input_ids = q_ids + a_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(q_ids) + a_ids + [tokenizer.eos_token_id]
    token_cnt = len(input_ids)
    return (input_ids, labels), token_cnt


def data_func_qa(data, tokenizer, args):
    save_dtype = np.int16 if args.save_dtype=="int16" else np.int32
    if args.merge_data:
        merged_data = []
        # 将每个数据放入第一个可以容纳它的位置，若没有能容纳的位置，就新建
        for input_ids, labels in data:
            fitted = False
            for i in range(len(merged_data)):
                if len(merged_data[i][0]) + len(input_ids) <= args.max_length:
                    merged_data[i][0] += input_ids
                    merged_data[i][1] += labels
                    fitted = True
                    break
            if not fitted:
                merged_data.append([input_ids, labels])
        
        ret_data = []
        # padding & change dtype
        for i in range(len(merged_data)):
            input_ids, labels = merged_data[i]
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
