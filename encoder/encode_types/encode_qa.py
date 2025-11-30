import heapq
import numpy as np


def encode_qa(tokenizer, conversation: dict):
    q_ids = tokenizer(conversation["q"])["input_ids"]
    a_ids = tokenizer(conversation["a"])["input_ids"]
    input_ids = q_ids + a_ids
    labels = [-100] * len(q_ids) + a_ids
    if input_ids[-1] != tokenizer.eos_token_id:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
    token_cnt = len(input_ids)
    return (input_ids, labels), token_cnt

def data_func_qa(data, tokenizer, args):
    save_dtype = np.int16 if args.save_dtype == "int16" else np.int32
    
    if args.merge_data:
        merged_data = []
        heapq.heapify(merged_data)  # 初始化空堆
        
        # 用优先队列模拟 best-fit 法求解装箱问题
        for input_ids, labels in data:
            if len(input_ids) >= args.max_length:
                continue  # 丢弃过长数据
            
            # 将新的数据尝试与堆中最短的合并
            merged = False
            if merged_data:
                # 取出堆顶，根据长度合并
                shortest_len, shortest_input_ids, shortest_labels = heapq.heappop(merged_data)
                if shortest_len + len(input_ids) <= args.max_length:
                    # 合并 input_ids 和 labels
                    shortest_input_ids += input_ids
                    shortest_labels += labels
                    new_len = len(shortest_input_ids)
                    # 重新压回堆中，按新长度排序
                    heapq.heappush(merged_data, (new_len, shortest_input_ids, shortest_labels))
                    merged = True
                else:
                    # 如果不能合并，将其放回堆中
                    heapq.heappush(merged_data, (shortest_len, shortest_input_ids, shortest_labels))
            
            # 如果无法合并，则创建一个新的条目
            if not merged:
                heapq.heappush(merged_data, (len(input_ids), input_ids, labels))
        
        ret_data = []
        # Padding 并调整数据类型
        for _, input_ids, labels in merged_data:
            pad_len = args.max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            input_ids = np.array(input_ids, dtype=save_dtype)
            labels = np.array(labels, dtype=save_dtype)
            ret_data.append((input_ids, labels))
        
        return ret_data
    
    else:
        ret_data = []
        # 非合并模式下进行常规的 padding 和类型转换
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