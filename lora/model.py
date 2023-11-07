from dataclasses import dataclass

import torch


@dataclass
class LoraConfig:
    r: int = 1
    alpha: int = 32
    dropout: float = 0.1
    target_module: str = "W_pack"


class LoraModel(torch.nn.Module):
    def __init__(self, base_model, config: LoraConfig):
        super().__init__()
        self.config = config
        self.model = self.inject_adapter(base_model)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor):
        return self.model.forward(input_ids, attention_mask)

    def inject_adapter(self, base_model):
        for name, module in base_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if not name.endswith(f".{self.config.target_module}"):
                    continue

                new_module = LoraLinear(
                    module, self.config.r, self.config.alpha, self.config.dropout
                )
                parent_name = ".".join(name.split(".")[:-1])
                parent_module = base_model.get_submodule(parent_name)
                # replace
                setattr(parent_module, self.config.target_module, new_module)

        for name, parameter in base_model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

        return base_model

    def count_trainable_parameters(self):
        trainable = 0
        total = 0
        for _, param in self.model.named_parameters():
            num_params = param.numel()
            total += num_params
            if param.requires_grad:
                trainable += num_params
        return trainable, total, f"{trainable/total:.6f}"


class LoraLinear(torch.nn.Linear):
    def __init__(self, layer, r, alpha, dropout):
        super().__init__(layer.in_features, layer.out_features)
        self.weight = layer.weight

        self.lora_dropout = torch.nn.Dropout(p=dropout)
        self.lora_A = torch.nn.Linear(layer.in_features, r, bias=False)
        self.lora_B = torch.nn.Linear(r, layer.out_features, bias=False)
        self.scale = alpha / r

    def forward(self, x: torch.Tensor):
        # weights not merged
        result = torch.nn.functional.linear(x, self.weight)
        result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scale
        return result
