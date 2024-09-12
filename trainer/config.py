from peft import LoraConfig, PromptTuningConfig, TaskType, PromptTuningInit

deepspeed_config = {
    "bf16": {
        "enabled": True,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas": [0.98, 0.999],
            "eps": 1e-9
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 1e-6,
            "warmup_max_lr": 1e-5,
            "warmup_num_steps": 1000
        }
    },
    "zero_optimization": {
        "stage": 1,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    "gradient_accumulation_steps": 16,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False,
    "steps_per_print": 100
}


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj",
        "gate_proj",
        "down_proj",
        "up_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)