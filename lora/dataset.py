import json
import random

import torch

from baichuan2.tokenizer import BaichuanTokenizer


def build_from_conversation(
    tokenizer,
    conversations: list[dict[str, str]],
    human=195,
    gpt=196,
    ignore=-100,
):
    input_ids = []
    labels = []

    # diff from https://github.com/baichuan-inc/Baichuan2/blob/10b6e6e220aee107e3206a4b9d23a414586588c4/fine-tune/fine-tune.py#L79C1-L86C58
    for i in conversations:
        value = tokenizer.encode(i["value"])
        if i["from"] == "human":
            input_ids += [human] + value
            labels += [ignore] * (len(value)+1)
        elif i["from"] == "gpt":
            input_ids += [gpt] + value
            labels += value + [tokenizer.eos_token_id]

    return input_ids, labels


#[{"id":0,"conversations":[{"from":"human","value":""},{"from":"gpt","value":""},...]},{"id":1...},...]
class ChatDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, tokenizer: BaichuanTokenizer, data_json_path: str, model_max_length: int, shuffle_dataset: bool
    ):
        self.tokenizer = tokenizer
        self.data_json = json.load(open(data_json_path, "r", encoding="utf-8"))
        if shuffle_dataset:
            random.shuffle(self.data_json)
        self.model_max_length = model_max_length

    def __len__(self):
        return len(self.data_json)

    def __iter__(self):
        for session in self.data_json:
            input_ids, labels = build_from_conversation(
                self.tokenizer, session["conversations"]
            )

            # padding or trim from beginning
            if len(input_ids) > self.model_max_length:
                input_ids = input_ids[0 : self.model_max_length]
            else:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * (
                    self.model_max_length - len(input_ids)
                )
            if len(labels) > self.model_max_length:
                labels = labels[0 : self.model_max_length]
            else:
                labels = labels + [-100] * (self.model_max_length - len(labels))

            input_ids = torch.LongTensor(input_ids)
            labels = torch.LongTensor(labels)
            mask = input_ids.ne(self.tokenizer.pad_token_id)
            yield input_ids.cuda(), labels.cuda(), mask.cuda()