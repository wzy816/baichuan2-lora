# baichuan2-lora

Fine-tuning baichuan2 13B base model using low-rank adaptation (LoRA).

## Feature

- baichuan2 code rewrite and lora implementation simplified
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

## Hyperparam Search

```bash
wandb offline

python3 -m lora.search \
    --project='baichuan2-lora_multiturn' \
    --vocab_file='/mnt/Baichuan2-13B-base/tokenizer.model' \
    --checkpoint_dir='/mnt/Baichuan2-13B-base' \
    --data_json_path='/mnt/baichuan2-lora/data/multiturn_chat_0.8M_reformat.json' \
    --output_dir='/mnt/baichuan2-lora_multiturn'

wandb sync --sync-all
```

## Train

```bash
wandb online

python3 -m lora.train \
    --project='baichuan2-lora_multiturn' \
    --vocab_file='/mnt/Baichuan2-13B-base/tokenizer.model' \
    --checkpoint_dir='/mnt/Baichuan2-13B-base' \
    --data_json_path='/mnt/baichuan2-lora/data/multiturn_chat_0.8M_reformat.json' \
    --config_path='/mnt/baichuan2-lora/config/multiturn_chat.yaml' \
    --output_dir='/mnt/baichuan2-lora_multiturn'
```

[viz notebook](hyperparam%20visualization.ipynb)

## Lora Model Inference

```bash

python3 -m lora.inference \
    --checkpoint_dir=/mnt/Baichuan2-13B-base \
    --lora_checkpoint_dir=/mnt/baichuan2-lora_multiturn/20231219_094052/step=556_loss=2.031 \
    --vocab_file=/mnt/Baichuan2-13B-base/tokenizer.model \
    --prompt='[{"from":"human","value":"写一篇介绍人工智能的科普文章。"},{"from":"gpt","value":"非常感谢您的需求，以下是关于人工智能的科普文章：\n人工智能（AI）是一种通过计算机技术模拟人类智能的技术，旨在创造一个能够感知、理解、学习、推理、适应和自我修复的智能系统。人工智能涉及多个领域，包括机器学习、自然语言处理和计算机视觉等。\n机器学习是使计算机自动执行任务的技术，并通过自我学习和优化来提高性能。自然语言处理是一种使计算机理解人类语言的技术，例如在自动翻译和语音识别中。计算机视觉利用人工智能技术来使计算机能够自动理解和解释数字图像或视频。\n虽然人工智能已经在许多领域得到应用，但目前仍存在挑战和限制，例如数据质量、算法偏见和安全性问题。但是，人工智能技术仍具有潜力，在很多方面能够帮助人类解决问题，例如自动驾驶、医疗诊断和能源管理等领域。"},{"from":"human","value":"从这篇文章中提取人工智能技术的优点和缺点。"}]' \
    --max_new_tokens=128

# 首先，人工智能可以帮助我们解决很多复杂的问题；其次，它也可以帮助我们更好地了解我们的世界并做出更好的决策。此外，由于机器学习的进步以及更多的数据和计算能力的可用性，未来将会有更多的人工智能的应用出现。然而，我们也应该意识到其潜在的威胁和挑战，如隐私和安全等问题。
```
