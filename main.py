from functools import partial
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import random
import os
from kronos import kronos_sharedQ

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(30 * x)

# H-SIREN: hyperbolic sine activation for first layer
class HSine(nn.Module):
    def forward(self, x):
        return torch.sin(torch.sinh(2 * x))

def positional_encoding(coords, L=10):
    freqs = 2.0 ** torch.linspace(0., L - 1, L).to(coords.device)
    encodings = coords[..., None] * freqs
    return torch.cat([torch.sin(encodings), torch.cos(encodings)], dim=-1).flatten(1)

class SIREN(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        input_dim = 2 * 10 * 2  # 2D coordinates * L=10 * (sin + cos)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            HSine(),
            nn.Linear(hidden_dim, hidden_dim),
            Sine(),
            nn.Linear(hidden_dim, hidden_dim),
            Sine(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, coords):
        x = positional_encoding(coords).to(coords.dtype)
        return self.net(x)

def train_single_image():
    # Set manual seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda")

    # Load single image (manually download an ImageNet1k sample)
    img = torchvision.io.read_image("/content/imagenet-sample-images/n01514668_cock.JPEG")
    img = torchvision.transforms.Resize((256, 256))(img)
    img = ((img.float() / 255.0) * 2) - 1

    # Setup coordinate grid
    x = torch.linspace(0, 256, 256)
    y = torch.linspace(0, 256, 256)
    xx, yy = torch.meshgrid(x, y)
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)

    # Initialize SIREN
    siren = SIREN().to(device)

    # Identify parameters for Nuon optimizer (≥2D), here picking multiple layers as examples
    kronos_params = [siren.net[2].weight, siren.net[4].weight,siren.net[6].weight,siren.net[8].weight]  # Flattened list of specific weights
    # print(nuon_params)

    # AdamW parameters: everything else
    adamw_params = [p for n, p in siren.named_parameters() if n not in ('net.2.weight', 'net.4.weight','net.6.weight','net.8.weight')]
    # print(adamw_params)

    # Ensure all parameters are accounted for
    assert len(kronos_params) + len(adamw_params) == len(list(siren.parameters()))


    # Assuming Muon is defined elsewhere
    optimizer = Kronos_sharedQ(
        kronos_params,
        lr=0.00035,
        lr_param=1.0,
        momentum=0.9,
        adamw_params=adamw_params,
        adamw_lr=3e-4,
        adamw_betas=(0.90, 0.95),
        adamw_wd=0
    )


    # Prepare target pixels
    pixels = img.reshape(-1, 3).to(device)

    # Learning-rate schedule
    def get_lr(it):
        return 1 - it / 5000

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    losses = []
    for i in range(5000):
        pred = siren(coords)
        loss = nn.MSELoss()(pred, pixels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        losses.append(loss.item())
        if i % 500 == 0:
            print(f"Loss at iteration {i}: {loss.item():.6f}")


    print(f"Loss at iteration {i}: {loss.item():.12f}")

    # Plot training loss
    plt.semilogy(losses)
    plt.ylabel('MSE')
    plt.xlabel('Iteration')
    plt.title('Training Loss')
    plt.show()

    # Inference
    with torch.no_grad():
        pred = siren(coords).float()
        pred_img = (pred.reshape(256, 256, 3).cpu() + 1) / 2

        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.imshow((img.reshape(256, 256, 3) + 1) / 2)
        plt.title('Original')

        plt.subplot(122)
        plt.imshow(pred_img)
        plt.title('SIREN Reconstruction')

        # plt.savefig('siren_reconstruction.jpeg')
        # plt.show()

    return siren, optimizer,losses

if __name__ == "__main__":
    trained_siren, nuon_optimizer, losses_shared = train_single_image()
