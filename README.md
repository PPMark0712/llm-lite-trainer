# llm-lite-trainer

## encoder

### 使用

配置环境

```text
创建并激活环境
conda create -n yyz_train python=3.11
conda activate yyz_train

安装cudatoolkit和cudatoolkit-dev
conda install nvidia/label/cuda-11.8.0::cuda-toolkit
conda install -c conda-forge cudatoolkit-dev=11.8

安装pytorch 2.4.0（从官网上复制的对应版本的指令）
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

安装flash attention，若无法执行，可以去掉后面的"--no-build-isolation"
pip install flash-attn --no-build-isolation

安装其他依赖包
pip install transformers deepspeed peft matplotlib
```


在任意目录下编写并运行run.sh脚本，例如：

```bash
cd /data1/yyz/projects/llm-lite-trainer/encoder

python /data1/yyz/projects/llm-lite-trainer/encoder/main.py \
    --tokenizer_path /data1/yyz/downloads/models/Qwen/Qwen2-7B \
    --data_loader_path /data1/yyz/projects/llm-lite-trainer/encoder/user_data_loader/alpaca.py \
    --corpus_name alpaca \
    --encode_type qa \
    --output_path output/debug_alpaca \
    --max_length 4096 \
    --merge_data \
    --save_dtype int32 \
```

参数说明

```text
--tokenizer_path str, tokenizer绝对路径 \
--data_loader_path str, 自定义data_loader.py绝对路径 \
--corpus_name str, 编码的语料名称，用于设置输出文件夹名称 \
--encode_type "qa"或"pretrain" \
--output_path str, 输出绝对路径，建议为/path/to/output/model_name/ \
--max_length int, 每条数据的长度（不得超过模型输入长度）
--merge_data store_true, 是否合并qa数据
--save_dtype 保存文件的数据类型，可选择"int32","int16"
--tokens_per_file 每个文件的有效token数量，默认5e8
```


用户需要自己写一个data_loader.py（文件名任意），并在其中实现data_loader()迭代器函数，用于迭代自定义的数据集，可以参考encoder/user_data_loader目录中的例子。

```py
# data_loader for import
def data_loader():
    yield ""
```


不同encode_type有不同返回格式：

```
pretrain: 返回一个str，表示一条预训练数据

qa: 返回一个带有"q"和"a"这两个key的dict，它们对应的value类型是str，表示问题和回答的文本
{
    "q": str,
    "a": str
}
```



### 编码文件格式

输出文件目录如下：

```text
output_path/
└── corpus_name/
    ├── corpus_0.bin
    ├── corpus_0.idx
    ├── corpus_1.bin
    ├── corpus_1.idx
    └── ...
```

其中的.bin文件是多个二进制数据对象，.idx文件中有一个数组（List[int]）对象，它存放了每一个对象在.bin文件中的偏移量，同时，这个数组的长度即为.bin文件中的对象数量

关于如何读取这些文件，可以参考trainer/dataset.py中的类。


## trainer

### 使用

1、在任意目录下编写并运行run.sh脚本，例如：

```bash
cd /data1/yyz/projects/llm-lite-trainer/trainer/

model_path=path/to/your/model
data_path=path/to/your/data
output_path=path/to/save/checkpoints
save_name=your_model_name  # for example: "Qwen2-7B-alpaca", "llama3-8B-Instruct-MetaMathQA"


deepspeed --include localhost:0,1,2,3 --master_port 12345 train.py \
    --model_path $model_path \
    --data_path $data_path \
    --output_path $output_path \
    --shuffle_data \
    --save_name $save_name \
    --max_epochs 1 \
    --save_steps 2000
```

参数说明

```text
# 训练参数
--model_path str, base模型路径
--max_epochs int, 最多训练的epoch数量
--max_steps int, 最多训练的step数量，一个step是dataloader的一轮迭代
--load_ckpt_path str, 默认None，加载的存档点路径
--data_path str, 数据路径根目录，会读取路径中所有的.bin和.idx文件（需使用本项目的encoder生成）
--deepspeed_config_path str, ds_config.json绝对路径，若不指定则使用项目中的config
--shuffle_data store_true, 是否随机打乱训练数据
--add_tokens str, nargs='+', 添加的特殊token，会修改模型嵌入层结构
# 输出参数
--output_path str, 输出绝对路径
--save_name str, 保存文件夹名称（建议为模型名+训练数据名），必须指定，同时还会自动生成时间戳
--save_steps int, 每多少步保存存档点
--save_epochs int, 每多少论保存存档点
--save_optimizer store_true, 保存优化器状态（若设置，存档点大小会扩大到7倍左右）
# lora参数
--use_lora store_true, 使用lora
--lora_config_path str, lora_config.json绝对路径，若不指定则使用项目中的config
```

