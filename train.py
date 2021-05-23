import sys
import random
import os

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torchvision import transforms

sys.path.append(".")
from utils import get_gen_loss, get_disc_loss


def train(
        dataset,
        device,
        gen,
        gen_opt,
        disc,
        disc_opt,
        adv_criterion,
        lambda_recon,
        recon_criterion,
        n_epochs,
        display_step,
        batch_size,
        model_name,
        save_model=True,
        cur_step=0
):
    writer_real = SummaryWriter(f'logs/logs_{model_name}/real')
    writer_fake = SummaryWriter(f'logs/logs_{model_name}/fake')
    writer_condition = SummaryWriter(f'logs/logs_{model_name}/condition')

    try:
        os.mkdir('saved_model_paths')
    except FileExistsError:
        pass

    train_dataset, val_dataset = random_split(
        dataset,
        [int(len(dataset) * 0.95), len(dataset) - int(len(dataset) * 0.95)]
    )

    dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # applies a horizontal flip on an image
    flip = torch.jit.script(torch.nn.Sequential(transforms.RandomHorizontalFlip(p=1)))

    mean_generator_loss = 0
    mean_discriminator_loss = 0

    for epoch in range(n_epochs):
        for image, _ in tqdm(dataloader, file=sys.stdout):
            # Input Setup
            condition = image[:, :, :, 256:].to(device)
            real = image[:, :, :, :256].to(device)

            # 50% chance of flipping the the images horizontally
            if random.random() > 0.5:
                real = flip(real)
                condition = flip(condition)

            # Update discriminator
            disc_opt.zero_grad()
            disc_loss = get_disc_loss(gen, disc, real, condition, adv_criterion)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Update generator
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(
                gen,
                disc,
                real,
                condition,
                adv_criterion,
                recon_criterion,
                lambda_recon,
            )
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the average loss
            mean_discriminator_loss += disc_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step

            # Visualization code
            if cur_step % display_step == 0:
                print()
                mean_val_loss = 0
                for val_image, _ in tqdm(val_dataloader, file=sys.stdout, position=0, leave=True):
                    condition = val_image[:, :, :, 256:].to(device)
                    real = val_image[:, :, :, :256].to(device)

                    with torch.no_grad():
                        gen_loss = get_gen_loss(
                            gen,
                            disc,
                            real,
                            condition,
                            adv_criterion,
                            recon_criterion,
                            lambda_recon,
                        )
                        mean_val_loss += gen_loss.item() / len(val_dataloader)

                print(
                    f"Epoch {epoch}: Step {cur_step}: "
                    f"Generator loss: {mean_generator_loss}, "
                    f"Generator Val Loss: {mean_val_loss}, "
                    f"Discriminator loss: {mean_discriminator_loss}, "
                )

                # Log with tensorboard
                with torch.no_grad():
                    fake = gen(condition)
                    img_grid_real = torchvision.utils.make_grid(real, normalize=True)
                    img_grid_condition = torchvision.utils.make_grid(condition, normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

                    writer_real.add_image("Real", img_grid_real, global_step=cur_step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=cur_step)
                    writer_condition.add_image("Condition", img_grid_condition, global_step=cur_step)

                mean_generator_loss = 0
                mean_discriminator_loss = 0

                if save_model and cur_step % 2000 == 0:
                    torch.save(
                        {'gen': gen.state_dict(),
                         'gen_opt': gen_opt.state_dict(),
                         'disc': disc.state_dict(),
                         'disc_opt': disc_opt.state_dict()
                         },
                        f"saved_model_paths/{model_name}_{cur_step}.pth"
                    )

                    if cur_step >= 8000:
                        return

            cur_step += 1
