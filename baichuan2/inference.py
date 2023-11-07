import gc
import os

import click
import torch
import torch.nn.functional as F
from baichuan2.model import BaichuanConfig, BaichuanForCausalLM
from baichuan2.tokenizer import BaichuanTokenizer
from tqdm import tqdm


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 5,
    top_p: float = 0.85,
):
    if top_k > 0:
        mask = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[mask] = -float("Inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float("Inf")
    return logits


@torch.inference_mode()
def generate(model, tokenizer, config, prompt, max_new_tokens):
    prompt_tokens = tokenizer.encode(prompt)
    tokens = (
        torch.full((1, len(prompt_tokens) + max_new_tokens), config.pad_token_id)
        .long()
        .cuda()
    )
    tokens[0, : len(prompt_tokens)] = torch.LongTensor(prompt_tokens).cuda()

    for i in tqdm(range(max_new_tokens)):
        end = i + len(prompt_tokens)
        start = max(0, end - config.model_max_length)
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
        if next_token == config.eos_token_id:
            t = tokens[0, len(prompt_tokens) : end].tolist()
            return t, tokenizer.decode(t)
    t = tokens[0, len(prompt_tokens) :].tolist()
    return t, tokenizer.decode(t)


@click.command()
@click.option("checkpoint_dir", "--checkpoint_dir", required=True)
@click.option("vocab_file", "--vocab_file", required=True)
@click.option("prompt", "--prompt", required=True)
@click.option("max_new_tokens", "--max_new_tokens", required=True, type=int)
def main(checkpoint_dir, vocab_file, prompt, max_new_tokens):
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)

    tokenizer = BaichuanTokenizer(vocab_file=vocab_file)
    config = BaichuanConfig()
    assert config.vocab_size == tokenizer.vocab_size()
    assert config.pad_token_id == tokenizer.pad_token_id
    assert config.bos_token_id == tokenizer.bos_token_id
    assert config.eos_token_id == tokenizer.eos_token_id

    model = BaichuanForCausalLM(config)

    shard_files = [
        os.path.join(checkpoint_dir, f"pytorch_model-0000{i+1}-of-00003.bin")
        for i in range(3)
    ]
    for f in tqdm(shard_files, desc="loading model shards"):
        assert os.path.exists(f)

        state_dict = torch.load(f, map_location="cuda:0")
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()

    answer_tokens, answer = generate(model, tokenizer, config, prompt, max_new_tokens)
    print(answer)


if __name__ == "__main__":
    main()
