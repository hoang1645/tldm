from typing import Callable, Dict, List
import torch
from torch import nn
from transformers import T5EncoderModel, AutoTokenizer

from models.norm import AdaLN
import math
from functools import reduce
from operator import mul


multiply = lambda x: reduce(mul, x)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = AdaLN(hidden_size, components=2)

    def forward(self, x, c):
        """
        x: input
        c: conditioning
        """
        shift, scale = self.adaLN_modulation(x, c)
        x = self.adaLN_modulation.modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps:torch.Tensor):
        half_dim = self.dim >> 1
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        # assert len(input_size) == 3, "Input size must be 3-dimensional"
        # self.pe = nn.Parameter(
        #     torch.sin(torch.arange(0, multiply(input_size)) % input_size[-1]) 
        #                        + torch.cos(torch.arange(input_size[-2])), requires_grad=False)

    def forward(self, x:torch.Tensor):
        input_size = x.size()
        pe = torch.sin(torch.arange(0, multiply(input_size)) % input_size[-1] / 10000).reshape(x.shape) + \
                             torch.cos(torch.arange(input_size[-2]) / 10000)
        pe = pe.to(x.device)

        x = x + pe
        return x
    

class TextConditioner(nn.Module):
    def __init__(self, model_name_or_path:str="google/flan-t5-small", model_type:Callable[..., nn.Module]=T5EncoderModel,
                 token_limit:int|None=None):
        super().__init__()
        # model_type output must support last_hidden_state
        # when in doubt, just pass AutoModel.
        # model_name_or_path must be a valid repo from HF Hub or a local path with the same structure.
        
        self.model = model_type.from_pretrained(model_name_or_path)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except ValueError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        self.token_limit = token_limit

    def forward(self, text:str|list[str]):
        if isinstance(text, str):
            text = [text]
        text = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=self.token_limit, truncation=True).input_ids.to(self.model.device)
        return self.model(text).last_hidden_state
    


class TextConditionerCollection(nn.Module):
    def __init__(self, model_kwargs: List[Dict], token_limit:int, freeze:bool=False):
        super().__init__()
        self.models = nn.ModuleList([TextConditioner(token_limit=token_limit, **kwargs) for kwargs in model_kwargs])
        if freeze:
            for model in self.models:
                model.requires_grad_(False)
    def forward(self, text:str|list[str]):
        if isinstance(text, str):
            text = [text]
        return torch.cat([model(text) for model in self.models], dim=-1)
    


class TextConditionEmbedding(nn.Module):
    def __init__(self, hidden_size, model_kwargs: List[Dict], token_limit:int, freeze:bool=True):
        super().__init__()
        self.text_conditioner = TextConditionerCollection(model_kwargs, token_limit, freeze)
        self.adapter = nn.LazyLinear(hidden_size, bias=True)
    def forward(self, x):
        x = self.text_conditioner(x)
        return self.adapter(x)
    


class Patchifier(nn.Module):
    def __init__(self, patch_size:int, in_channels:int, out_channels:int):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x:torch.Tensor):
        return self.conv(x)