import torch
from torch import nn
from torch.nn.utils import spectral_norm

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
        print("x" , x.shape)
        lstm_out = self.lstm(x)
        print("lstm_out" , lstm_out.shape)

        noise = torch.rand(
            (x.shape[0], self.noise_dimension),
            dtype=torch.float32,
            device=lstm_out.device,
        )
        # TODO: add time window to orders and noise
        lstm_noise = torch.cat((lstm_out, noise), dim=1)
        print("noise" , noise.shape)
        print("lstm_noise" , lstm_noise.shape)
        out = self.main(lstm_noise)
        return out

class PrintModule(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x
        
class Discriminator(nn.Module):
    def __init__(
        self,
        history_length=50,
        n_features=9,
        noise_dimension=50,
        lstm_hidden_dim=200,
        dropout=0.5,
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
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        lstm_out = self.lstm(x)
        out = self.main(lstm_out)
        return out