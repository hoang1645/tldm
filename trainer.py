import torch
from diffusers import AutoencoderKL, AutoencoderTiny
from models.diffusion import LDM
from typing import Callable

from rich.progress import BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn, Progress
from rich.console import Console

class Trainer:
    def __init__(self, model:LDM, optimizer:torch.optim.Optimizer,
                 scheduler:torch.optim.lr_scheduler.LambdaLR|None,
                 train_dataset:torch.utils.data.Dataset,
                 val_dataset:torch.utils.data.Dataset,
                 num_loader_workers:int,
                 batch_size:int,
                 n_epochs:int,
                 autocast:bool=False,
                 loss_log_callback:Callable=None,
                 start_epoch:int=0,
                 start_step:int=0,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.train_dataset = train_dataset
        # self.val_dataset = val_dataset
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size, shuffle=True, num_workers=num_loader_workers 
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_loader_workers
        )
        self.n_epochs = n_epochs
        self.console = Console()

        self.pbar = Progress(
            TextColumn("[green]{task.description} / epoch {task.fields[current_epoch]}/{task.fields[total_epoch]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("||"),
            TimeRemainingColumn(),
            TextColumn("loss = {task.fields[loss]:.4}"), console=self.console,
            transient=True
        )

        self.autocast = autocast
        self.loss_log_callback = loss_log_callback
        self.step = start_step
        self.epoch = start_epoch

    def train_single_step(self, step:int, img:torch.Tensor, caption:list[str]):
        batch_size = img.shape[0]
        if isinstance(self.model.autoencoder, AutoencoderKL):
            latent = self.model.autoencoder(img).latent_dist.sample()
        else:
            latent = self.model.autoencoder(img).latents

        self.optimizer.zero_grad()

        cond = self.model.encode_text(caption).to(img.device)
        timestep_tensor = torch.randint(1, self.model.n_diffusion_steps,
                                        size=(batch_size, ),
                                        device=img.device)
        latent, noise = self.model.forward_diffusion(latent, timestep_tensor)

        with torch.amp.autocast("cuda", torch.bfloat16, enabled=self.autocast):
            predicted = self.model.forward(latent, timestep_tensor.float().to(img.device), cond)

        loss = torch.nn.functional.mse_loss(predicted, noise)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        if self.loss_log_callback and step % 50 == 0:
            self.loss_log_callback("train_loss", loss.item(), step)
    
        return loss.item()

    def train_single_epoch(self, epoch:int):
        task = self.pbar.add_task("Training", total=self.train_loader, loss=0.,
                                  current_epoch=epoch, total_epoch=self.n_epochs)
        self.model.train()
        for img, caption in self.train_loader():
            loss = self.train_single_step(self.step, img, caption)
            self.step += 1
            self.pbar.update(task, advance=1, loss=loss, current_epoch=epoch, total_epoch=self.n_epochs)

        self.pbar.stop_task(task)

        
    @torch.no_grad()
    def eval(self, epoch:int):
        task = self.pbar.add_task("Validating", total=len(self.val_loader), loss=0.,
                                  current_epoch=epoch, total_epoch=self.n_epochs)
        
        self.model.eval()
        losses = []
        for img, caption in self.val_loader:
            batch_size = img.shape[0]
            if isinstance(self.model.autoencoder, AutoencoderKL):
                latent = self.model.autoencoder(img).latent_dist.sample()
            else:
                latent = self.model.autoencoder(img).latents

            cond = self.model.encode_text(caption).to(img.device)
            timestep_tensor = torch.randint(1, self.model.n_diffusion_steps,
                                            size=(batch_size, ),
                                            device=img.device)
            latent, noise = self.model.forward_diffusion(latent, timestep_tensor)

            with torch.amp.autocast("cuda", torch.bfloat16, enabled=self.autocast):
                predicted = self.model.forward(latent, timestep_tensor.float().to(img.device), cond)

            loss = torch.nn.functional.mse_loss(predicted, noise)
            self.pbar.update(task, advance=1, loss=loss.item(), current_epoch=epoch, total_epoch=self.n_epochs)
            losses.append(loss)
        try:
            self.console.print(
                f"Evaluation complete: epoch {epoch}/{self.n_epochs}, loss = {sum(losses) / len(losses)}"
            )
        except ZeroDivisionError:
            self.console.print(
                f"Evaluation complete: epoch {epoch}/{self.n_epochs}, loss = ???"
            )
        
        self.pbar.stop_task(task)
        if self.loss_log_callback:
            self.loss_log_callback("val_loss", sum(losses) / len(losses), epoch)

    def train(self):
        for epoch in range(self.epoch, self.n_epochs):
            self.train_single_epoch(epoch)