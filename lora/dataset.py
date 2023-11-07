import json
import random

import torch


class BelleChatDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, data_json_path, model_max_length):
        self.tokenizer = tokenizer
        self.data_json = json.load(open(data_json_path, "r"))
        random.shuffle(self.data_json)
        self.model_max_length = model_max_length

    def __len__(self):
        return len(self.data_json)

    def __iter__(self):
        for session in self.data_json:
            input_ids = []
            labels = []
            human = [195]
            gpt = [196]
            eos = self.tokenizer.eos_token_id
            pad = self.tokenizer.pad_token_id
            ignore = [-100]

            for msg in session["conversations"]:
                value = self.tokenizer.encode(msg["value"])
                if msg["from"] == "human":
                    input_ids += human + value
                    labels += [eos] + ignore * len(value)
                elif msg["from"] == "gpt":
                    input_ids += gpt + value
                    labels += ignore + value
                else:
                    raise Exception("unknow conversation role")

            input_ids = input_ids + [eos]
            if len(input_ids) > self.model_max_length:
                input_ids = input_ids[0 : self.model_max_length]
            else:
                input_ids = input_ids + [pad] * (self.model_max_length - len(input_ids))

            labels = labels + [eos]
            if len(labels) > self.model_max_length:
                labels = labels[0 : self.model_max_length]
            else:
                labels = labels + ignore * (self.model_max_length - len(labels))

            input_ids = torch.LongTensor(input_ids)
            labels = torch.LongTensor(labels)
            mask = input_ids.ne(pad)
            yield input_ids.cuda(), labels.cuda(), mask.cuda()
