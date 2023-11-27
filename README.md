# baichuan2-lora

Fine-tuning baichuan2 13B base model using low-rank adaptation (LoRA).

## Feature

- baichuan2 code rewrite and lora implementation simplified
- fix alibi mask [issue](https://github.com/baichuan-inc/Baichuan2/issues/225)
- remove peft, deepspeed and transformers dependency
- training on a single GPU 80G

## Prepare Env

```bash
conda create -n baichuan2-lora python=3.10
conda activate baichuan2-lora
cd baichuan2-lora
pip install -r requirements.txt
```

## Download Weights File and Tokenizer File

from [huggingface](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/tree/main)

```bash
tree /mnt/Baichuan2-13B-base/

/mnt/Baichuan2-13B-base/
├── pytorch_model-00001-of-00003.bin
├── pytorch_model-00002-of-00003.bin
├── pytorch_model-00003-of-00003.bin
└── tokenizer.model

# Other files are optional.
```

## Base Model Inference

```bash
python3 -m baichuan2.inference \
    --checkpoint_dir=/mnt/Baichuan2-13B-base \
    --vocab_file=/mnt/Baichuan2-13B-base/tokenizer.model \
    --prompt='登鹳雀楼->王之涣\n夜雨寄北->' \
    --max_new_tokens=128

# 李商隐\n泊秦淮->杜牧\n望庐山瀑布->李白\n春晓->孟浩然\n绝句->杜甫\n出塞->王昌龄\n凉州词->王之涣\n芙蓉楼送辛渐->王昌
# not optimized, can be slow for long answer
```

## Train with Belle Chat

```bash
python3 -m lora.train \
    --project='baichuan2-lora_belle' \
    --vocab_file='/mnt/Baichuan2-13B-base/tokenizer.model' \
    --checkpoint_dir='/mnt/Baichuan2-13B-base' \
    --data_json_path='./data/belle_chat_ramdon_10k.json' \
    --config_yaml_path='./config/belle_chat.yaml' \
    --output_dir='/mnt/baichuan2-lora_belle'
```

## Hyperparam Search

```bash
python3 -m lora.search \
    --project='baichuan2-lora_belle' \
    --vocab_file='/mnt/Baichuan2-13B-base/tokenizer.model' \
    --checkpoint_dir='/mnt/Baichuan2-13B-base' \
    --data_json_path='/mnt/baichuan2-lora/data/belle_chat_ramdon_10k.json' \
    --output_dir='/mnt/baichuan2-lora_belle'

wandb sync --sync-all
```
[viz notebook](hyperparam visualization.ipynb)

## Lora Model Inference

```bash
python3 -m lora.inference \

```
