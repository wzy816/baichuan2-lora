from flask import Flask, jsonify, make_response, request
from tqdm import tqdm

app = Flask(__name__)

import torch
import torch.nn.functional as F
from baichuan2.inference import top_k_top_p_filtering
from baichuan2.model import BaichuanConfig
from baichuan2.tokenizer import BaichuanTokenizer
from lora.dataset import build_from_conversation
from lora.model import LoraModel
from lora.util import load_base_model, load_lora_config

torch.set_default_device("cuda")

vocab_file = "/mnt/Baichuan2-13B-base/tokenizer.model"
checkpoint_dir = "/mnt/Baichuan2-13B-base"
lora_checkpoint_dir = (
#    "/mnt/baichuan2-lora_multiturn/20231225_101753/step=1955_loss=1.773"
#    "/mnt/baichuan2-lora_marketing/20240219_143746/step=309_loss=0.001"
#	"/mnt/baichuan2-lora_marketing/20240220_181148/step=264_loss=1.805"
#	"/mnt/baichuan2-lora_marketing/20240221_154022/step=221_loss=0.031"
	"/mnt/baichuan2-lora_marketing/20240225_110213/step=1177_loss=0.014"
)

tokenizer = BaichuanTokenizer(vocab_file=vocab_file)

model_config = BaichuanConfig()
base_model = load_base_model(checkpoint_dir, torch.float32, model_config)

lora_config = load_lora_config(f"{lora_checkpoint_dir}/lora_config.json")
lora_model = LoraModel(base_model=base_model, config=lora_config)
state = torch.load(f"{lora_checkpoint_dir}/weights.pt", map_location="cuda:0")
lora_model.load_state_dict(state, strict=False)

max_new_tokens = 2048
model_max_length = 4096


@app.route("/chat", methods=["POST"])
def chat_api():
    try:
        conversations = request.json
        return generate(
            lora_model,
            tokenizer,
            conversations,
            model_max_length,
            max_new_tokens,
        ), {"Content-Type": "text/event-stream"}

    except Exception as e:
        print(str(e))
        return b"error", 500


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
        x = tokens[:, start:end].cuda()
        logits = model.forward(x)
        next_token_logits = logits[:, -1, :]

        # repetition penalty
        repetition_penalty = 1.05 # 1.2
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

        if next_token != tokenizer.eos_token_id:
            yield tokenizer.decode(next_token.tolist())
        else:
            return


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
