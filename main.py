from models.diffusion import LDM
from torch import nn
import torch
from torchinfo import summary
import yaml

with open("configs/model.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)["model"]


def parse_module_and_class(module_class:str):
    module, cls = module_class.rsplit(".", 1)
    return getattr(__import__(module, fromlist=[cls]), cls)

#-------------------vae-------------------
vae = parse_module_and_class(config["vae"]["type"]).from_pretrained(config["vae"]["model_name_or_path"], subfolder=config["vae"]["subfolder"])
#-------------------text------------------
text_model_kwargs = config["text_encoder"]
for kwargs in text_model_kwargs:
    kwargs["model_type"] = parse_module_and_class(kwargs["model_type"])
token_limit = config["token_limit"]
model_kwargs = config["diffusion"]
activation_kwargs = model_kwargs["activation"]["kwargs"] or {}

model_kwargs["activation"] = parse_module_and_class(model_kwargs["activation"]["type"])



model = LDM(vae, **model_kwargs, text_model_kwargs=text_model_kwargs, token_limit=token_limit, **activation_kwargs)

summary(model)

condition = ["Hello, I am a condition"]
x =  torch.randn(1, 3, 256, 256).cuda()



latent = model.autoencoder.encode(x).latents
c = model.encode_text(condition).cuda()
print(c.shape)
t = torch.randint(0, 1000, (1,)).cuda()
out = model.forward(latent, t, c)
print(out.shape)
print(model.backward_diffusion_sampling(condition, 1000, 1).size)

# from transformers import AutoTokenizer
# from datasets.datasets import DBParquetDataset
# from tqdm import tqdm
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
# dset = DBParquetDataset("/mnt/e/db")
# for _, text in tqdm(dset):
#     tokens = tokenizer(text, return_tensors="pt").input_ids
#     assert tokens.numel() <= 256, (text, tokens.numel())