import os
import sys

import torch
import torchvision
from torch import nn
from torchvision import transforms

sys.path.append(".")
from model import UNet, Discriminator
from train import train
from utils import initialize_model

# HYPERPARAMETERS
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

ADV_CRITERION = nn.BCEWithLogitsLoss()
RECON_CRITERION = nn.L1Loss()

INPUT_DIM = 3
REAL_DIM = 3
TARGET_SHAPE = 256

GEN_LR = 0.0002
DISC_LR = 0.0002

LAMBDA_RECON = 99

NUM_EPOCHS = 75
BATCH_SIZE = 24
HIDDEN_CHANNELS = 32

DISPLAY_STEP = 500

# DATA SETUP
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def main(path_to_dataset, model_name):
    # finds the latest saved path if it exists
    try:
        saved_model_paths = [path for path in os.listdir('saved_model_paths') if model_name in path]
        saved_model_path = sorted(saved_model_paths)[-1]
        pretrained = True
        cur_step = int(saved_model_path[saved_model_path.find('_') + 1:-4])
        saved_model_path = f'saved_model_paths/{saved_model_path}'
        print(f'Loading from {saved_model_path}')
    except (IndexError, FileNotFoundError):
        pretrained = False
        saved_model_path = None
        cur_step = 0

    # MODEL SETUP
    gen = UNet(INPUT_DIM, REAL_DIM, hidden_channels=HIDDEN_CHANNELS).to(DEVICE)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=GEN_LR)

    disc = Discriminator(INPUT_DIM + REAL_DIM).to(DEVICE)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=DISC_LR)

    initialize_model(gen, gen_opt, disc, disc_opt, device=DEVICE,
                     pretrained=pretrained, saved_model_path=saved_model_path)

    # DATA SETUP
    dataset = torchvision.datasets.ImageFolder(path_to_dataset, transform=TRANSFORM)

    # TRAINING
    train(
        dataset=dataset,
        device=DEVICE,
        gen=gen,
        gen_opt=gen_opt,
        disc=disc,
        disc_opt=disc_opt,
        adv_criterion=ADV_CRITERION,
        lambda_recon=LAMBDA_RECON,
        recon_criterion=RECON_CRITERION,
        n_epochs=NUM_EPOCHS,
        display_step=DISPLAY_STEP,
        batch_size=BATCH_SIZE,
        model_name=model_name,
        save_model=True,
        cur_step=cur_step
    )


if __name__ == '__main__':
    for hair_color in ['black', 'blonde', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red']:
        main(path_to_dataset=f'./dataset/{hair_color}_hair/', model_name=hair_color)
