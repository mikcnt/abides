import torch
from torch import nn
import pytorch_lightning as pl
from typing import *
import pandas as pd


# useful module to create a view of a tensor inside a sequential
class ViewModule(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view(*self.args)


# use LSTM and take last output
class LstmSequence(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_dim,
        dropout=0.0,
        num_layers=1,
        bidirectional=False,
        batch_first=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

    def forward(self, x):
        out = self.lstm(x)[0]
        out_forward = out[:, -1, : self.hidden_dim]
        if not self.bidirectional:
            return out_forward
        else:
            out_reverse = out[:, 0, self.hidden_dim :]
            return torch.cat((out_forward, out_reverse), dim=1)


class Generator(nn.Module):
    def __init__(
        self,
        n_features=9,
        noise_dimension=50,
        lstm_hidden_dim=200,
        dropout=0.5,
        n_generated_orders=1,
    ):
        super().__init__()
        self.noise_dimension = noise_dimension
        self.lstm = LstmSequence(input_size=n_features, hidden_dim=lstm_hidden_dim)
        self.main = nn.Sequential(
            nn.Linear(
                in_features=lstm_hidden_dim + noise_dimension,
                out_features=(n_features - 5) * n_generated_orders * 100,
            ),
            nn.BatchNorm1d((n_features - 5) * 100),
            nn.ReLU(),
            ViewModule(-1, 100, n_generated_orders, n_features - 5),
            nn.Upsample(scale_factor=2),
            nn.Dropout(dropout),
            nn.Upsample(scale_factor=2),
            self._tconv_block(100, 32, 3, 1, 1),
            self._tconv_block(32, 16, 3, 1, 1),
            self._tconv_block(16, 8, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(8, 1, 3, 1, 1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            ViewModule(-1, 1, n_features - 5),
        )

    def _tconv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        lstm_out = self.lstm(x)
        noise = torch.rand(
            (x.shape[0], self.noise_dimension),
            dtype=torch.float32,
            device=lstm_out.device,
        )
        lstm_noise = torch.cat((lstm_out, noise), dim=1)
        out = self.main(lstm_noise)
        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        history_length=50,
        n_features=9,
        noise_dimension=50,
        lstm_hidden_dim=200,
        n_generated_orders=1,
    ):
        super().__init__()
        self.lstm = LstmSequence(input_size=n_features, hidden_dim=lstm_hidden_dim)
        self.main = nn.Sequential(
            nn.Linear(in_features=lstm_hidden_dim, out_features=256 * 3),
            nn.BatchNorm1d(256 * 3),
            nn.ReLU(),
            ViewModule(-1, 3, 16, 16),
            self._conv_block(3, 128, 3, 1, 1),
            self._conv_block(128, 64, 3, 1, 1),
            self._conv_block(64, 32, 3, 1, 1),
            ViewModule(-1, 32 * 16 * 16),
            nn.Linear(in_features=32 * 16 * 16, out_features=4),
        )

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        lstm_out = self.lstm(x)
        out = self.main(lstm_out)
        return out


class WGANGP(pl.LightningModule):
    def __init__(
        self,
        hparams,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        # Generator
        self.G = Generator()

        # Discriminator
        self.D = Discriminator()

        # Validation outputs & logs
        self.outputs_dir = "outputs/"
        self.logged_results = {"real": [], "generated": []}
        self.trade_columns = ["size", "price", "direction", "time_diff"]
        self.columns_real = ["real_" + f for f in self.trade_columns]
        self.columns_fake = ["fake_" + f for f in self.trade_columns]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.G(x)

    # def train_dataloader(self) -> DataLoader:
    #     train_loader = DataLoader(
    #         TradesDataset(self.hparams["msg_path"]),
    #         batch_size=self.hparams["batch_size"],
    #         drop_last=True,
    #         shuffle=False,
    #         pin_memory=True,
    #     )
    #     return train_loader

    # def val_dataloader(self) -> DataLoader:
    #     val_dataloader = DataLoader(
    #         TradesDataset(self.hparams["msg_path"]),
    #         batch_size=self.hparams["batch_size"],
    #         drop_last=True,
    #         shuffle=False,
    #         pin_memory=True,
    #     )
    #     return val_dataloader

    def configure_optimizers(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Optimizers
        optimizer_G = torch.optim.Adam(
            self.G.parameters(),
            lr=self.hparams["lr"],
            betas=(self.hparams["b1"], self.hparams["b2"]),
        )
        optimizer_D = torch.optim.Adam(
            self.D.parameters(),
            lr=self.hparams["lr"],
            betas=(self.hparams["b1"], self.hparams["b2"]),
        )
        return (
            {"optimizer": optimizer_G, "frequency": 1},
            {"optimizer": optimizer_D, "frequency": self.hparams["critic_iterations"]},
        )

    def _gradient_penalty(self, x: torch.Tensor, y: torch.Tensor):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((x.shape[0], 1, 1)).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (
            (alpha * x + ((1 - alpha) * y)).requires_grad_(True).to(self.device)
        )
        with torch.backends.cudnn.flags(enabled=False):
            d_interpolates = self.D(interpolates)
        fake = torch.ones(d_interpolates.shape).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.shape[0], -1).to(self.device)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def training_step(
        self, batch: List[torch.Tensor], batch_idx: int, optimizer_idx: int
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Unpack the batch
        trade_features, orderbook_features = batch

        # concatenate trade features and orderbook features
        real_trades = torch.cat((trade_features, orderbook_features), dim=-1)

        # get last trade (which represents the real trade to be compared with the generated one)
        last_orderbook = orderbook_features[:, -1:, :]

        # gen_input are the first `sequence_length` features
        gen_input = real_trades[:, :-1, :]

        # generate trade with generator
        generated_trade = self.G(gen_input)
        generated_trade_all = torch.cat((generated_trade, last_orderbook), dim=-1)

        # concatenate
        fake_trades = torch.cat((gen_input, generated_trade_all), dim=1)

        # The first optimizer is to train the generator
        if optimizer_idx == 0:
            disc_fake = self.D(fake_trades)
            loss_G = -torch.mean(disc_fake)

            self.log("Loss G.", loss_G, prog_bar=True, on_step=False, on_epoch=True)
            return loss_G

        # The second optimizer is to train the discriminator
        elif optimizer_idx == 1:
            disc_real = self.D(real_trades)
            disc_fake = self.D(fake_trades)
            loss_D = (
                -torch.mean(disc_real)
                - torch.mean(disc_fake)
                + self.hparams["lambda_gp"]
                * self._gradient_penalty(real_trades, fake_trades)
            )
            self.log("Loss D.", loss_D, prog_bar=True, on_step=False, on_epoch=True)
            return loss_D

        raise RuntimeError("There is an error in the optimizers configuration.")

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        # Unpack the batch
        trade_features, orderbook_features = batch

        # concatenate trade features and orderbook features
        real_trades = torch.cat((trade_features, orderbook_features), dim=-1)

        # get last trade (which represents the real trade to be compared with the generated one)
        last_orderbook = orderbook_features[:, -1:, :]

        # gen_input are the first `sequence_length` features
        gen_input = real_trades[:, :-1, :]

        # generate trade with generator
        real_trade = trade_features[:, -1, :]
        generated_trade = self.G(gen_input).reshape(trade_features.shape[0], 4)
        return {"real_trade": real_trade, "generated_trade": generated_trade}

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        real = []
        generated = []
        for x in outputs:
            real.append(x["real_trade"])
            generated.append(x["generated_trade"])

        real = torch.cat(real, dim=0).squeeze()
        generated = torch.cat(generated, dim=0).squeeze()
        real = pd.DataFrame(real.detach().cpu().numpy())
        generated = pd.DataFrame(generated.detach().cpu().numpy())
        trades_df = pd.concat((real, generated), axis=1)
        trades_df.columns = self.columns_real + self.columns_fake

        # outputs directory
        epoch = str(self.current_epoch).zfill(3)
        current_outputs_dir = self.outputs_dir + epoch
        os.makedirs(current_outputs_dir, exist_ok=True)
        # save `csv` containing validation outputs
        trades_df.to_csv(current_outputs_dir + f"/results_{epoch}.csv")
        # save plots
        wandb_plots = self._plot_results(trades_df, epoch, datadir=current_outputs_dir)
        self.log_dict({f"Distribution plots epoch {epoch}": wandb_plots})
        return

    def _plot_results(self, trades_df: pd.DataFrame, epoch: str, datadir: str):
        """Save distribution plots."""
        # generate and save distribution plot for each column
        images = []
        for col in self.trade_columns:
            fig = plt.figure()
            real_col = "real_" + col
            fake_col = "fake_" + col
            sns.distplot(trades_df[real_col], label="Real")
            sns.distplot(trades_df[fake_col], label="Fake")
            plt.legend()
            plt.title(f"{col} epoch {epoch}")
            plt.xlabel(col)
            plt.savefig(datadir + f"/{col}.png")
            images.append(wandb.Image(fig))
            plt.clf()
        return images