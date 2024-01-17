# torch
import torch
from torch import nn
# dataset + dataloader
from datasets.datasets import PixivDataset
from torch.utils.data import DataLoader
# loggers
from torch.utils.tensorboard import SummaryWriter
from comet_ml import Experiment
# self-defined utilities
from models.unet import UNet
from models.diffuser import LDM
from models.autoencoder.autoencoders_cnn import AutoencoderVQ
# progress bar
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn, \
    SpinnerColumn
# argument parser
import argparse
# model summary
from torchinfo import summary
# autocast
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
# other utilities
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T
from generation_evaluate.metrics import FID
import os
# typing
from typing import Callable


def train(model: AutoencoderVQ, timesteps: int,
          reconstruction_loss_fn: nn.Module | Callable[..., torch.Tensor],
          optimizer: torch.optim.Optimizer,
          train_dataloader: DataLoader,
          n_epoch: int = 100, start_from_epoch: int = 0, start_step: int = 0,
          with_autocast: bool = True, log_comet: bool = False,
          comet_api_key: str = None, comet_project_name: str = None, ckpt_save_path:str=None):
    if ckpt_save_path is None:
        ckpt_save_path = '.'
    # Prepare model
    model = model.train()
    # initialize loggers
    tensorboard_logger = SummaryWriter()  # log saved at run/
    comet_logger = None
    if log_comet:
        if comet_api_key is None or comet_project_name is None:
            raise ValueError("API key and project name required when using Comet logger")
        comet_logger = Experiment(comet_api_key, comet_project_name, log_code=True, auto_param_logging=True)

    # Prepare gradient scaler, if autocast is used
    a_grad_scaler = None
    if with_autocast:
        a_grad_scaler = GradScaler()
        

    # stablizing training on autoencoder
    a_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4000, args.lr / 10)

    # initialize training step
    training_step = start_step

    # console for rich
    console = Console()
    for epoch in range(start_from_epoch, n_epoch):
        # save losses for logging to console
        losses = []
        rlosses = []
        # progress bar
        pbar = Progress(TextColumn("[green]Epoch {}/{}".format(epoch, n_epoch)), BarColumn(), MofNCompleteColumn(),
                        TimeElapsedColumn(), TextColumn("||"), TimeRemainingColumn(),
                        TextColumn("loss = {task.fields[loss]:.4}, reconstruction loss = {task.fields[rloss]:.4}"), console=console,
                        transient=True)

        task = pbar.add_task("", total=len(train_dataloader), loss=0.0, rloss=.0)
        pbar.start()
        # train
        for img in train_dataloader:
            # size of batch in dataloader
            batch_size = img.shape[0]
            img = img.float().to(model.device)
            x0, idx, vqloss = model.encode(img)
            

            optimizer.zero_grad()

            if with_autocast:
                with autocast():
                    x0 = model.decode(x0)
                    r_loss = reconstruction_loss_fn(x0, img) + .25 * vqloss
                a_grad_scaler.scale(r_loss).backward()
                a_lr_scheduler.step()
                a_grad_scaler.step(optimizer)
                a_grad_scaler.update()
            else:
                x0 = model.autoencoder.decode(x0)
                r_loss = reconstruction_loss_fn(x0, img) + .25 *vqloss
                r_loss.backward()
                a_lr_scheduler.step()
                optimizer.step()
            
            # log shit
            if (training_step + 1) % 50 == 0:
                tensorboard_logger.add_scalar('r_loss', r_loss, training_step)
                if log_comet:
                    comet_logger.log_metric('r_loss', r_loss, training_step)


            rlosses.append(r_loss.item())
            pbar.update(task, advance=1, loss=0., rloss=r_loss.item())
            training_step += 1

        # summarize and save checkpoint
        console.print(f"Epoch {epoch}, average loss = {sum(losses) / len(losses)}, reconstruction loss = {sum(rlosses) / len(losses)}.")
        state_dict = {"epoch": epoch + 1, "step": training_step + 1, "state_dict": model.state_dict(),
                      
                      "autoencoder_optim": optimizer.state_dict()}
        console.print("Saving checkpoint...")
        torch.save(state_dict, os.path.join(ckpt_save_path,
                                            "checkpoints/AE-epoch={epoch}_rloss={r_loss:.4}.pth".format(
            epoch=epoch,  r_loss=r_loss.item())))
        console.print(f"Checkpoint saved at [i]{os.path.join(ckpt_save_path, 'checkpoints/AE-epoch={epoch}_rloss={r_loss:.4}.pth'.format(epoch=epoch, r_loss=r_loss.item()))}[/i]")
        pbar.stop()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=False, default=None)
    # parser.add_argument('--generator_size', type=str, choices=['XS', 'S', 'B', 'L'], required=False, default='S')
    parser.add_argument('--image_size', type=int, choices=[128, 256, 1024], required=False, default=256)
    # parser.add_argument('--normalization', type=str, choices=['identity', 'batch', 'sync-batch', 'layer'],
    #                     required=False, default='batch')
    # parser.add_argument('--eps', type=float, required=False, default=1e-6)
    # parser.add_argument('--discriminator_type', type=str, choices=['style-gan', 'resnet18', 'resnet34'],
    #                     default='style-gan', required=False)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, help="sanity check")
    parser.add_argument('--lr', type=float, required=False, default=2e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, required=False, default=4, help='batch size')
    parser.add_argument('--beta1', type=float, required=False, default=.9, help='adam optimization algorithm\'s β1')
    parser.add_argument('--beta2', type=float, required=False, default=.99, help='adam optimization algorithm\'s β2')
    parser.add_argument('--autocast', action=argparse.BooleanOptionalAction, required=False, help='use automatic type casting')
    parser.add_argument('--from_ckpt', type=str, required=False, default=None, help='load model from checkpoint at specified path')
    parser.add_argument('--infer', action=argparse.BooleanOptionalAction, required=False, help='generate image instead of training')
    parser.add_argument('--epoch', type=int, required=False, default=100, help='number of training epochs')
    parser.add_argument('--evaluate_fid', action=argparse.BooleanOptionalAction, 
                        help='evaluate FID wrt evaluation dataset (must specify --evaluation_target). ignored if --infer is not on')
    parser.add_argument('--evaluation_target', type=str, required=False, default=None, 
                        help='evaluation target, used in conjunction with --evaluate_fid. ignored if --infer is not on')
    parser.add_argument('--num_gen', type=int, required=False, default=100, 
                        help='how many images to generate for FID eval, used in conjunction with --evaluate_fid. ignored if --infer is not on')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, help='torch.compile(model, mode=\'reduce-overhead\') (requires torch>=2.0 and linux kernel)')
    parser.add_argument('--reset_optimizers', action=argparse.BooleanOptionalAction)
    parser.add_argument('--save_ckpt', type=str, help='where to save checkpoints. defaults to the current directory.', required=False, default=None)
    # parser.add_argument('--update_discriminator_every_n_steps', type=int, required=False, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    unet = UNet(in_chan=32, out_chan=32, embed_dim=128, n_attn_heads=8, dim_head=64, conv_init_chan=160, chan_mults=(1,2,4,4))
    model = AutoencoderVQ(64, quant_dim=32, codebook_size=8192)
    model = model.cuda()
    # model = LDM(unet, autoencoder, 256)
    d_optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    
    if args.compile:
        model = torch.compile(model, backend='inductor', mode='reduce-overhead')
    
    print("Model initialized")

    summary(model, verbose=1)
    if args.debug:
        x = torch.randn(1, 3, args.image_size, args.image_size).to(model.device)
        t = torch.randint(1, 1000, size=(1,)).to(model.device)
        print(model(x)[0].shape) 
        exit(0)
    start = 0
    step = 0

    if args.from_ckpt is not None:
        dicts = torch.load(args.from_ckpt)
        model.load_state_dict(dicts['state_dict'], strict=False)
        start = dicts['epoch']
        step = dicts['step']
        if not args.reset_optimizers:
            d_optim.load_state_dict(dicts['optim'])

        print("Loaded from checkpoint", args.from_ckpt)

    elif not args.infer:
        print("No checkpoint provided. Training from scratch")

    else:
        raise ValueError("Inference requires a model checkpoint.")


    if args.dataset_path is None: raise ValueError("Training expected a dataset path.")
    

    noise_loss = nn.MSELoss()
    reconstruction_loss = nn.MSELoss()
    train_dataloader = DataLoader(
        PixivDataset(args.dataset_path, imageSize=256,
                    return_original=False, transforms=T.Lambda(lambda t: (t * 2) - 1)),
        batch_size=args.batch_size, shuffle=True
    )

    

    train(model, 1000, reconstruction_loss, d_optim, 
            train_dataloader, with_autocast=args.autocast,
        n_epoch=args.epoch, start_from_epoch=start, start_step=step, ckpt_save_path=args.save_ckpt)

            