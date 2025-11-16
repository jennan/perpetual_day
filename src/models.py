import torch
import torch.nn as nn
import lightning as L
import diffusers
from torch.optim.lr_scheduler import CosineAnnealingLR


class CNN(L.LightningModule):
    def __init__(
        self,
        chan_in,
        chan_out,
        chan_latent=10,
        learning_rate=1e-4,
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

        self.loss_function = nn.functional.mse_loss
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

    def predict(self, features, targets_shape):
        features = torch.from_numpy(features).to(self.device)
        preds = self(features).to("cpu").detach().numpy()
        return preds


class UNet(L.LightningModule):
    def __init__(
        self,
        chan_in,
        chan_out,
        sample_size,
        learning_rate=1e-4,
        eta_min: float = 1e-6,
        T_max: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.unet = diffusers.UNet2DModel(
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

        self.loss_function = nn.functional.mse_loss
        self.learning_rate = learning_rate
        self.eta_min = eta_min
        self.T_max = T_max

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
        scheduler = CosineAnnealingLR(optimizer, eta_min=self.eta_min, T_max=self.T_max)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict(self, features, targets_shape):
        features = torch.from_numpy(features).unsqueeze(0).to(self.device)
        preds = self(features).squeeze(0).to("cpu").detach().numpy()
        return preds


class DiffusionModel(L.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float = 1e-4,
        eta_min: float = 1e-6,
        T_max: int = 5,
    ):
        super().__init__()
        self.model = model
        self.scheduler = diffusers.schedulers.DDPMScheduler()
        self.loss_function = nn.functional.mse_loss
        self.learning_rate = learning_rate
        self.eta_min = eta_min
        self.T_max = T_max

    def forward(self, x, t, conds):
        net_input = torch.cat((x, conds), 1)
        return self.model(net_input, t).sample

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
        scheduler = CosineAnnealingLR(optimizer, eta_min=self.eta_min, T_max=self.T_max)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict(self, features, targets_shape):
        x = torch.randn(*targets_shape).unsqueeze(0).to(self.device)
        y = torch.from_numpy(features).unsqueeze(0).to(self.device)
        for i, t in enumerate(self.scheduler.timesteps):
            with torch.no_grad():
                residual = self(x, t, y)
            x = self.scheduler.step(residual, t, x).prev_sample
        preds = x.squeeze(0).to("cpu").detach().numpy()
        return preds


class DiffUNet2D(DiffusionModel):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        eta_min: float = 1e-6,
        T_max: int = 5,
        **model_kwargs,
    ):
        model = diffusers.UNet2DModel(**model_kwargs)
        super().__init__(model, learning_rate, eta_min, T_max)
        self.save_hyperparameters()
