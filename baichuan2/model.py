import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BaichuanConfig:
    hidden_size: int = 5120
    intermediate_size: int = 13696
    num_hidden_layers: int = 40
    num_attention_heads: int = 40
    model_max_length: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    vocab_size: int = 125696
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, epsilon=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size))
        self.epsilon = epsilon

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class MLP(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class BaichuanAttention(torch.nn.Module):
    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.model_max_length

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        self.W_pack = torch.nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=False
        )
        self.o_proj = torch.nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        bsz, q_len, _ = hidden_states.size()

        proj = self.W_pack(hidden_states)
        proj = (
            proj.unflatten(-1, (3, self.hidden_size))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
        )
        query_states = (
            proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key_states = (
            proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class BaichuanLayer(torch.nn.Module):
    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = BaichuanAttention(config=config)
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class BaichuanModel(torch.nn.Module):
    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.n_head = config.num_attention_heads
        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = torch.nn.ModuleList(
            [BaichuanLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

    def alibi_mask(self, num_heads, seq_len):
        position_point = torch.arange(seq_len)[None, :] - torch.arange(seq_len)[:, None]

        s = (2**8) ** (1 / num_heads)
        m = (
            torch.tensor([1 / s ** (i + 1) for i in range(num_heads)])
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        bias = m * position_point.unsqueeze(0)
        bias = bias + torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        )
        return bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_length = input_ids.shape

        inputs_embeds = self.embed_tokens(input_ids)
        alibi_mask = self.alibi_mask(
            self.n_head, seq_length
        )  # (n_head, seq_len, seq_len)

        if attention_mask is None:
            attention_mask = alibi_mask
        else:
            # attention_mask equals input_ids.ne(pad token id)
            mask = attention_mask.to(torch.float32)  # (batch_size, seq_len)
            mask = 1 - torch.tril(mask[:, :, None] * mask[:, None, :], diagonal=0)
            mask = mask.masked_fill(mask > 0, float("-inf"))
            mask = (
                mask.unsqueeze(1).expand(batch_size, 1, seq_length, seq_length).cuda()
            )
            attention_mask = mask + alibi_mask

        hidden_states = inputs_embeds
        for _, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
            )
        hidden_states = self.norm(hidden_states)

        return hidden_states


class NormHead(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((vocab_size, hidden_size)))

    def forward(self, hidden_states):
        norm_weight = torch.nn.functional.normalize(self.weight)
        return torch.nn.functional.linear(hidden_states, norm_weight)


class BaichuanForCausalLM(torch.nn.Module):
    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.model = BaichuanModel(config)
        self.lm_head = NormHead(config.hidden_size, config.vocab_size)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = self.lm_head(hidden_states)
        return logits
