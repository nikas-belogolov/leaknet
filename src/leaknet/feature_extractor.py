from torch import nn, Tensor

class ConvBlock(nn.Module):
    """Convolutional module: Conv1D + BatchNorm + (optional) ReLU."""
    def __init__(
        self,
        n_in_channels: int,
        n_out_channels: int,
        kernel_size: int,
        padding_mode: str = "replicate",
        include_relu: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv1d(
                in_channels=n_in_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding="same",
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(num_features=n_out_channels),
        ]
        if include_relu:
            layers.append(nn.ReLU())
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_block(x)
        return out

class FCNFeatureExtractor(nn.Module):
    def __init__(self, n_in_channels, padding_mode: str = "replicate"):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.instance_encoder = nn.Sequential(
            ConvBlock(n_in_channels, 16, 8, padding_mode=padding_mode),
            ConvBlock(16, 32, 5, padding_mode=padding_mode),
            ConvBlock(32, 16, 3, padding_mode=padding_mode),
        )
        
        self.output_dim = self.instance_encoder[-1].conv_block[0].out_channels
        
    def forward(self, x: Tensor):
        x = self.instance_encoder(x)
        return x