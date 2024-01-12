import json
import os

import click
import torch
import torch.nn.functional as F
from tqdm import tqdm

from baichuan2.inference import top_k_top_p_filtering
from baichuan2.model import BaichuanConfig
from baichuan2.tokenizer import BaichuanTokenizer
from lora.dataset import build_from_conversation
from lora.model import LoraModel
from lora.util import load_base_model, load_lora_config


@torch.inference_mode()
def generate(model, tokenizer, conversations, model_max_length, max_new_tokens):
    prompt_tokens, labels = build_from_conversation(tokenizer, conversations)
    prompt_tokens += [196]
    tokens = (
        torch.full((1, len(prompt_tokens) + max_new_tokens), tokenizer.pad_token_id)
        .long()
        .cuda()
    )
    tokens[0, : len(prompt_tokens)] = torch.LongTensor(prompt_tokens).cuda()

    for i in tqdm(range(max_new_tokens)):
        end = i + len(prompt_tokens)
        start = max(0, end - model_max_length)
        x = tokens[:, start:end]
        logits = model.forward(x)
        next_token_logits = logits[:, -1, :]

        # repetition penalty
        repetition_penalty = 1.2
        score = torch.gather(next_token_logits, 1, x)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        next_token_logits.scatter_(1, x, score)

        # temperature
        next_token_scores = next_token_logits / 0.3

        # top k top p
        next_token_scores = top_k_top_p_filtering(next_token_scores)

        probs = F.softmax(next_token_scores, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

        tokens[0, end] = next_token
        if next_token == tokenizer.eos_token_id:
            t = tokens[0, len(prompt_tokens) : end].tolist()
            return t, tokenizer.decode(t)

    t = tokens[0, len(prompt_tokens) :].tolist()
    return t, tokenizer.decode(t)


@click.command()
@click.option("vocab_file", "--vocab_file", required=True)
@click.option("checkpoint_dir", "--checkpoint_dir", required=True)
@click.option("lora_checkpoint_dir", "--lora_checkpoint_dir", required=True)
@click.option("prompt", "--prompt", required=True)
@click.option("max_new_tokens", "--max_new_tokens", required=True, type=int)
def main(vocab_file, checkpoint_dir, lora_checkpoint_dir, prompt, max_new_tokens):
    tokenizer = BaichuanTokenizer(vocab_file=vocab_file)

    model_config = BaichuanConfig()
    base_model = load_base_model(checkpoint_dir, torch.float32, model_config)

    assert os.path.exists(lora_checkpoint_dir)
    lora_config = load_lora_config(f"{lora_checkpoint_dir}/lora_config.json")
    lora_model = LoraModel(base_model=base_model, config=lora_config)
    state = torch.load(f"{lora_checkpoint_dir}/weights.pt", map_location="cuda:0")
    lora_model.load_state_dict(state, strict=False)

    conversations = json.loads(prompt)
    answer_token, answer = generate(
        lora_model,
        tokenizer,
        conversations,
        4096,  # model_config.model_max_length
        max_new_tokens,
    )
    print(answer)


if __name__ == "__main__":
    main()
