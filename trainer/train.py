import argparse, os, json, random, datetime
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import deepspeed
from peft import get_peft_model, PeftModel, LoraConfig

from config import lora_config, deepspeed_config
from dataset import TorchMultiFileBinaryDataset
from draw_loss import draw_loss


def print0(*args, **kwargs):
    """只在主进程print"""
    if torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def setup_distributed_environment(local_rank):
    """配置分布式训练环境"""
    if local_rank != -1:  # 使用分布式训练
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",  # 使用环境变量指定初始化方法。这意味着PyTorch会自动从环境变量中寻找必要的设置，如主机地址和端口号，以及进程的排名和总数。
            rank=local_rank,  # 设置当前进程的排名。
            world_size=torch.cuda.device_count(),  # 设置进程组中的进程总数，这里使用的是当前节点上可用的CUDA设备数
        )
    else:  # 单卡训练
        device = torch.device("cuda")
    deepspeed.init_distributed()
    return device


def initialize_model(device, lora_config, args):
    """加载和初始化模型"""
    print0("Loading model...")
    
    if args.load_ckpt_path and not args.use_lora:
        model_path = os.path.join(args.load_ckpt_path, args.ckpt_path, f"step_{args.load_ckpt_step}")
    else:
        model_path = args.model_path
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).to(device)

    if args.use_lora:
        if args.load_ckpt_path:
            load_ckpt_path = os.path.join(args.load_ckpt_path, args.ckpt_path, f"step_{args.load_ckpt_step}")
            model = PeftModel.from_pretrained(model, load_ckpt_path, is_trainable=True)
        else:
            print0("Using LoRa: Training from scratch")
            model = get_peft_model(model, lora_config)
    elif args.add_tokens:
        num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': args.add_tokens})
        model.resize_token_embeddings(len(tokenizer))  # 调整 embed 层,使其能够适应新的 token 数量
        embedding_layer = model.get_input_embeddings()  # 获取 embedding 层
        with torch.no_grad():
            new_token_indices = range(len(tokenizer) - num_added_tokens, len(tokenizer))
            for token_index in new_token_indices:
                embedding_layer.weight[token_index].uniform_(-0.1, 0.1)  # 均匀分布初始化
    return model, tokenizer


def prepare_dataloader(deepspeed_config, device, args):
    """准备数据加载器"""
    print0("Loading dataset...")
    train_dataset = TorchMultiFileBinaryDataset(args.data_path, device)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.local_rank != -1 else None
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=deepspeed_config["train_micro_batch_size_per_gpu"],
        num_workers=4,
    )
    return train_dataloader


def train_model(model, tokenizer, train_dataloader, ds_config, args):
    """模型训练循环"""
    engine, _, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=model.parameters(),
    )

    step = 0
    losses = []
    begin_epoch = 1  # 第一个epoch编号（加载存档点时有变化）
    end_epoch = args.max_epochs if args.max_epochs else (args.max_steps - 1) // len(train_dataloader) + 1
    begin_epoch_step = 0  # 加载存档点时，第一个epoch需要从第几个batch开始

    # 根据存档点来更新初始epoch和batch
    if args.load_ckpt_path:
        ckpt_path = os.path.join(args.load_ckpt_path, args.ckpt_path, f"{args.save_name}_{args.load_ckpt_step}")
        loss_fn = os.path.join(ckpt_path,"loss_list.json")
        with open(loss_fn, "r") as f:
            losses = json.load(f)
        begin_epoch = args.load_ckpt_step // len(train_dataloader) + 1
        begin_epoch_step = args.load_ckpt_step % len(train_dataloader)
        step = args.load_ckpt_step
    
    # 初始化进度条
    if dist.get_rank() == 0:    
        if args.max_steps:
            total_train_steps = args.max_steps - ((begin_epoch - 1) * len(train_dataloader) + begin_epoch_step)
        else:
            total_train_steps = (args.max_epochs - begin_epoch + 1) * len(train_dataloader)
        pbar = tqdm(total=total_train_steps, ncols=95)
        
        # 加载存档点batch的进度条
        if args.load_ckpt_path:
            skip_pbar = tqdm(total=begin_step, desc="Loading checkpoint", ncols=90)

    # 训练过程
    for epoch in range(begin_epoch, end_epoch + 1):
        begin_step = 0 if epoch > begin_epoch else begin_epoch_step
    
        for batch_id, batch in enumerate(train_dataloader):
            # 加载存档点batch
            if batch_id < begin_step:
                if epoch == begin_epoch and args.load_ckpt_path and dist.get_rank() == 0:
                    skip_pbar.update(1)
                continue
            if epoch == begin_epoch and args.load_ckpt_path and dist.get_rank() == 0:
                skip_pbar.close()

            # 前向传播，计算loss，反向传播
            loss = engine(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                use_cache=False,  # 训练过程不用 Attention 层 KV cache
            ).loss
            engine.backward(loss)
            engine.step()
            step += 1
            losses.append(loss.item())

            # 更新训练进度条
            if dist.get_rank() == 0:
                pbar.update()
                pbar.set_description(f"epoch:{epoch},batch:{batch_id + 1}/{len(train_dataloader)},loss:{np.mean(losses[-200:]):.4f}")

            # 根据save_steps保存存档点
            if args.save_steps and step % args.save_steps == 0:
                save_checkpoint(engine, tokenizer, step, losses, args)
            
            # 根据max_steps终止训练
            if step >= args.max_steps:
                break
        
        # 根据save_epochs保存存档点
        if args.save_epochs and epoch % args.save_epochs == 0:
            save_checkpoint(engine, tokenizer, step, losses, args)
        
        # 根据max_steps终止训练
        if step >= args.max_steps:
            break
    
    # 判断是否需要保存最后一个存档点
    if args.save_steps and args.max_steps % args.save_steps != 0:
        save_checkpoint(engine, tokenizer, step, losses, args)
    if args.save_epochs and epoch % args.save_epochs != 0:
        save_checkpoint(engine, tokenizer, step, losses, args)

    if dist.get_rank() == 0:
        pbar.close()


def save_checkpoint(engine, tokenizer, step, losses, args):
    # 保存模型和训练损失
    ckpt_path = os.path.join(args.save_path, args.ckpt_path, f"{args.save_name}_{step}")
    os.makedirs(ckpt_path, exist_ok=True)

    # 保存模型
    if args.use_lora:
        if args.local_rank != -1:  # 是分布式训练环境
            dist.barrier()  # 阻塞当前进程, 直到所有其他进程也调用了 dist.barrier(), 才会释放所有进程
        if torch.distributed.get_rank() == 0 or args.local_rank == -1:  # 主进程或非分布式训练环境
            engine.save_pretrained(ckpt_path)  
        if args.local_rank != -1:
            dist.barrier()
    else:
        engine.save_16bit_model(ckpt_path)  # 保存模型
        with open(os.path.join(ckpt_path, 'config.json'), 'w') as f:  # 保存config
            print(json.dumps(engine.module.config.to_dict(), indent=4), file=f)
        tokenizer.save_pretrained(ckpt_path)  # 保存tokenizer
    
    # 保存损失函数
    loss_file_name = os.path.join(ckpt_path, "loss_list.json")
    with open(loss_file_name, "w") as f:
        print(json.dumps(losses), file=f)

    draw_loss(ckpt_path)


def set_seed(seed):
    """设置随机数种子, 保证结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize(args):
    set_seed(args.seed)

    # load config
    with open(args.deepspeed_config_path, "r") as f:
        deepspeed_config = json.load(f)
    lora_config = None

    if args.load_ckpt_path:
        with open(args.lora_config_path, "r") as f:
            lora_config = LoraConfig(**json.load(f))
        args.save_path = args.load_ckpt_path
        file_list = os.listdir(os.path.join(args.load_ckpt_path, args.ckpt_path))
        file_list.sort(key=lambda x:int(x.split("_")[-1]))
        args.load_ckpt_step = int(file_list[-1].split("_")[-1])
        with open(os.path.join(args.load_ckpt_path, "train_config_0.json"), "r") as f:
            initial_config = json.load(f)
            initial_num_gpus = initial_config["args"]["num_gpus"]
            assert torch.cuda.device_count() == initial_num_gpus, "num_gpus can't change when loading ckpt!"
    else:
        t = datetime.datetime.now()
        args.save_path = os.path.join(args.output_path, f"{t.year}-{t.month:02d}-{t.day:02d}_{t.hour:02d}-{t.minute:02d}_{args.save_name}")
        args.load_ckpt_step = 0
        
    os.makedirs(args.save_path, exist_ok=True)
    config_fn = os.path.join(args.save_path, f"train_config_{args.load_ckpt_step}.json")
    with open(config_fn, "w") as f:
        config_show = {
            "args": {
                "num_gpus": torch.cuda.device_count(),
                **vars(args),
            },
            "deepspeed_config": deepspeed_config,
        }
        if args.use_lora:
            config_show.update({"lora_config": lora_config})

        print(json.dumps(config_show, indent=4), file=f)
    return deepspeed_config, lora_config

def get_args():
    """获得参数"""
    parser = argparse.ArgumentParser()
    # train params
    parser.add_argument("--max_epochs", type=int, required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=19260817)
    parser.add_argument("--load_ckpt_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, help="the root folder of your data")
    parser.add_argument("--deepspeed_config_path", type=str, required=True)
    parser.add_argument("--lora_config", type=str, default=None)
    # save params
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_epochs", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, default="ckpt")
    parser.add_argument("--save_name", type=str, required=True)
    # lora params
    parser.add_argument("--use_lora", action="store_true")
    # finetune params
    parser.add_argument("--add_tokens", nargs='+', default=None)
    # distribute params
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    assert bool(args.save_steps) != bool(args.save_epochs), "Specify exactly one of --save_steps or --save_epochs"
    if not args.load_lora:
        assert bool(args.model_path) or bool(args.load_ckpt_path), "Specify --model_path or --load_ckpt_path to define the base model."
    else:
        assert args.lora_config, "Specify --lora_config_path when --use_lora is set"
        assert args.add_tokens is None, "Do not specify --add_tokens when --use_lora is set."
        assert args.model_path, "Specify --model_path when --use_lora is set."
    if args.save_steps is not None and args.save_steps <= 0:
        raise ValueError("--save_steps must be greater than 0")
    if args.save_epochs is not None and args.save_epochs <= 0:
        raise ValueError("--save_epochs must be greater than 0")
    return args


def main():
    args = get_args()
    deepspeed_config, lora_config = initialize(args)
    device = setup_distributed_environment(args.local_rank)
    model, tokenizer = initialize_model(device, lora_config, args)
    train_dataloader = prepare_dataloader(deepspeed_config, device, args)
    train_model(model, tokenizer, train_dataloader, deepspeed_config, args)


if __name__ == "__main__":
    main()
