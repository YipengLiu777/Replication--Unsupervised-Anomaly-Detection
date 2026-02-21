import torch
import torch.nn as nn

class CAE_TableI_64(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.fc_enc = nn.Linear(64 * 8 * 8, latent_dim)  # IMPORTANT: no ReLU

        self.fc_dec = nn.Sequential(nn.Linear(latent_dim, 64 * 8 * 8), nn.ReLU(inplace=True))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1), nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(inplace=True))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1), nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(16, 3, 3, padding=1)

    def encode(self, x):
        h = self.conv1(x); h = self.conv2(h); h = self.conv3(h)
        return self.fc_enc(h.flatten(1))

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 64, 8, 8)
        h = self.deconv1(h); h = self.deconv2(h); h = self.deconv3(h)
        return torch.sigmoid(self.out_conv(h))

    def forward(self, x):
        z = self.encode(x)
        xrec = self.decode(z)
        return xrec, z

class CAE_TableI_224(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.flat_dim = 64 * 28 * 28
        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)  # IMPORTANT: no ReLU

        self.fc_dec = nn.Sequential(nn.Linear(latent_dim, self.flat_dim), nn.ReLU(inplace=True))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1), nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(inplace=True))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1), nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(16, 3, 3, padding=1)

    def encode(self, x):
        h = self.conv1(x); h = self.conv2(h); h = self.conv3(h)
        return self.fc_enc(h.flatten(1))

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 64, 28, 28)
        h = self.deconv1(h); h = self.deconv2(h); h = self.deconv3(h)
        return torch.sigmoid(self.out_conv(h))

    def forward(self, x):
        z = self.encode(x)
        xrec = self.decode(z)
        return xrec, z

def get_model(out_size: int, latent_dim: int):
    if out_size == 64:
        return CAE_TableI_64(latent_dim)
    if out_size == 224:
        return CAE_TableI_224(latent_dim)
    raise ValueError("out_size must be 64 or 224")