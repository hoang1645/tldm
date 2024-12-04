from models.diffusion import LDM
import torch
from torchinfo import summary
import yaml
import json
from argparse import ArgumentParser, BooleanOptionalAction

from trainer import Trainer
from datasets.datasets import DBParquetDataset
from torch.utils.tensorboard.writer import SummaryWriter


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", "-d", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=False, default="configs/model.yaml")
    parser.add_argument("--n_epochs", type=int, default=400)
    parser.add_argument("--batch_size", "-bsz", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", type=str, default="[0.9,0.99]")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--test_run", "-t", action=BooleanOptionalAction)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
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
        print(kwargs)
    token_limit = config["token_limit"]
    model_kwargs = config["diffusion"]
    activation_kwargs = model_kwargs["activation"]["kwargs"] or {}

    model_kwargs["activation"] = parse_module_and_class(model_kwargs["activation"]["type"])



    model = LDM(vae, **model_kwargs, text_model_kwargs=text_model_kwargs, token_limit=token_limit, **activation_kwargs,
                scheduler=config["scheduler"], n_backward_steps=50)

    summary(model, depth=4, verbose=1)

    if args.test_run:
        condition = ["Hello, I am a condition"]
        x =  torch.randn(1, 3, 256, 256).cuda()

        latent = model.autoencoder.encode(x).latents
        c = model.encode_text(condition).cuda()
        print(c.shape)
        t = torch.randint(0, 1000, (1,)).cuda()
        out = model.forward(latent, t, c)
        print(out.shape)
        model.backward_diffusion_sampling(condition, 1000, 1)
        exit(0)

    if args.ckpt:
        model.load_state_dict(
            torch.load(args.ckpt)
        )
    
    train_set = DBParquetDataset(args.dataset_path, split=0, imageSize=256, resize_rate=0.05)
    val_set = DBParquetDataset(args.dataset_path, split=1, resize_rate=0.05, imageSize=256)

    betas = json.loads(args.betas)
    optim = torch.optim.AdamW(model.parameters(), args.lr, betas)

    logger = SummaryWriter()
    trainer = Trainer(model, optim, None, train_set, val_set, 4, args.batch_size, args.n_epochs,
                      loss_log_callback=logger.add_scalar)

    trainer.train()
