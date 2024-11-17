from typing import Callable
import torch
import torch.nn as nn
from xformers.components import MultiHeadDispatch
from xformers.components.attention import build_attention
from models.norm import AdaLN



class MemoryEfficientAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = build_attention({
            "name": "scaled_dot_product",
            "dropout": dropout,
            "causal": False,  # Set to True if using causal attention
        })
        self.multihead = MultiHeadDispatch(
            dim_model=embed_dim,
            num_heads=num_heads,
            attention=self.attention,
        )

    def forward(self, x, k=None, v=None, attention_mask=None):
        return self.multihead(x, k, v, attention_mask)



class FFN(nn.Module):
    def __init__(self, model_dim:int, hidden_dim:int, dropout:float=0.,
                 activation:Callable[..., nn.Module]=nn.ReLU, **activation_kwargs):
        super().__init__()
        self.lin1 = nn.Linear(model_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, model_dim, bias=False)

        self.act = activation(**activation_kwargs)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x:torch.Tensor):
        x = self.lin1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x
    


class TransformerLayer(nn.Module):
    def __init__(self, model_dim:int, hidden_dim, n_heads, dropout, activation, **activation_kwargs):
        super().__init__()
        self.ffn = FFN(model_dim, hidden_dim, dropout, activation, **activation_kwargs)
        self.attn = MemoryEfficientAttention(model_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.adaptive = AdaLN(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor, conditioning:torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaptive(x, conditioning)
        attn_out = self.attn(self.adaptive.modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.dropout(attn_out) * gate_msa

        ffn_out = self.ffn(self.adaptive.modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + self.dropout(ffn_out) * gate_mlp
        return x





# # Example usage:
# if __name__ == "__main__":
#     model = MemoryEfficientTransformer(
#         num_layers=6,
#         embed_dim=512,
#         num_heads=8,
#         feedforward_dim=2048,
#         dropout=0.1
#     )

#     dummy_input = torch.randn(10, 32, 512)  # (sequence_length, batch_size, embed_dim)
#     output = model(dummy_input)
#     print(output.shape)  # Should output: (10, 32, 512)