import torch
from torch import nn

from models.transformer import TransformerLayer
from models.transformer.components import FinalLayer, TimestepEmbedding, PositionalEncoding, Patchifier
from einops import rearrange



class DiffusionTransformer(nn.Module):
    def __init__(self, patch_size:int, channels:int, num_layers:int,
                 model_dim:int, hidden_dim, n_heads, 
                 dropout, activation, **activation_kwargs):
        super().__init__()
        self.transformer = nn.ModuleList([
            TransformerLayer(model_dim, hidden_dim, n_heads, dropout, activation, **activation_kwargs)
            for _ in range(num_layers)
        ])
        self.patchifier = Patchifier(patch_size, channels, model_dim)
        self.final_layer = FinalLayer(model_dim, patch_size, channels)
        self.timestep_embedding = TimestepEmbedding(model_dim)
        self.positional_encoding = PositionalEncoding()
        self.embed_proj = nn.Linear(channels, model_dim)

        self.channels = channels
        self.patch_size = patch_size

    def forward(self, x, timesteps, c):
        """
        x: input (latent code + noise)
        c: conditioning
        timesteps: timesteps

        returns: noise
        """
        x = self.positional_encoding(x) # (batch, channels, w, h)
        x = self.patchifier(x) # (batch, model_dim, h // patch_size, w // patch_size)
        _, _, h, _ = x.shape
        x = rearrange(x, "b d h w -> b (h w) d") # (batch, h // patch_size * w // patch_size, model_dim)
        # timestep
        time_emb = self.timestep_embedding(timesteps) # (batch, model_dim)
        # shape of c: (batch, num_tokens, model_dim)
        c = c + time_emb[:, None, :]
        for layer in self.transformer:
            x = layer(x, c)

        x = self.final_layer(x, c) # (batch, w // patch_size * h // patch_size, patch_size * patch_size * channels)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, p1=self.patch_size, p2=self.patch_size, c=self.channels)
        return x
    

    def forward_with_cfg(self, x, t, c, cfg_scale):
        """
        yanked straight from the official code
        """

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, c, t)
        
        eps, rest = model_out[:, :self.channels], model_out[:, self.channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    

if __name__ == "__main__":
    model = DiffusionTransformer(
        patch_size=2,
        channels=4,
        num_layers=3,
        model_dim=512,
        hidden_dim=2048,
        n_heads=8,
        dropout=0.1,
        activation=nn.ReLU
    )

    dummy_input = torch.randn(4, 4, 32, 32)  # (batch_size, channels, width, height)
    timesteps = torch.arange(4)
    conditioning = torch.randn(4, 77, 512) 
    output = model(dummy_input, timesteps, conditioning)
    print(output.shape)  # Should output: (4, 4, 32, 32)