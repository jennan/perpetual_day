import torch
import torch.nn as nn
from diffusers import schedulers

import lightning as L


class CNN(L.LightningModule):
    def __init__(
        self,
        chan_in,
        chan_out,
        chan_latent=10,
        learning_rate=3e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=chan_in,
                out_channels=chan_latent,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=chan_latent,
                out_channels=chan_out,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

        self.loss_function = nn.functional.l1_loss
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        features, targets = batch
        outputs = self(features)
        loss = self.loss_function(outputs, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, targets = batch
        outputs = self(features)
        loss = self.loss_function(outputs, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}


class UNet(L.LightningModule):
    def __init__(
        self,
        chan_in,
        chan_out,
        sample_size=64,
        learning_rate=3e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize UNet2DModel
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=chan_in,
            out_channels=chan_out,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

        # Loss and optimizer settings
        self.loss_function = nn.functional.l1_loss
        self.learning_rate = learning_rate

    def forward(self, x):
        # dummy timestep value as we are not using diffusion
        timestep = torch.tensor([0], device=x.device)
        return self.unet(x, timestep).sample

    def training_step(self, batch, batch_idx):
        features, targets = batch
        outputs = self(features)
        loss = self.loss_function(outputs, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, targets = batch
        outputs = self(features)
        loss = self.loss_function(outputs, targets)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}


class DiffusionModel(L.LightningModule):
    def __init__(self, model, learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.scheduler = schedulers.DDPMScheduler()
        self.loss_function = nn.functional.l1_loss
        self.learning_rate = learning_rate

    def forward(self, x, t, conds):
        net_input = torch.cat((x, conds), 1)
        return self.unet(net_input, t).sample

    def training_step(self, batch, batch_idx):
        features, targets = batch

        noise = torch.randn_like(targets)
        steps = torch.randint(
            self.scheduler.config.num_train_timesteps,
            (targets.size(0),),
            device=self.device,
        )
        noisy_targets = self.scheduler.add_noise(targets, noise, steps)
        pred = self(noisy_targets, steps, features)

        loss = self.loss_function(pred, noise)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, targets = batch

        noise = torch.randn_like(targets)
        steps = torch.randint(
            self.scheduler.config.num_train_timesteps,
            (targets.size(0),),
            device=self.device,
        )
        noisy_targets = self.scheduler.add_noise(targets, noise, steps)
        pred = self(noisy_targets, steps, features)

        loss = self.loss_function(pred, noise)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}
