import matplotlib.pyplot as plt
import torch
# import torch.nn.functional as F
from torch import nn
from torchvision.utils import make_grid


# def get_one_hot_labels(labels, n_classes):
#     return F.one_hot(labels, num_classes=n_classes)
#
#
# def get_one_hot_matrix(label):
#     hair_color = get_one_hot_labels(label, 14)
#     hair_color = hair_color[:, :, None, None].repeat(1, 1, 256, 256)
#     one_hot_matrix = hair_color.float()
#
#     return one_hot_matrix


def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    fakes = gen(condition)
    preds = disc(fakes, condition)

    adv_loss = adv_criterion(preds, torch.ones_like(preds))
    rec_loss = recon_criterion(real, fakes)

    gen_loss = adv_loss + lambda_recon * rec_loss

    return gen_loss


def get_disc_loss(gen, disc, real, condition, adv_criterion):
    with torch.no_grad():
        fake = gen(condition)

    fake_pred = disc(fake.detach(), condition)
    real_pred = disc(real, condition)

    disc_fake_loss = adv_criterion(fake_pred, torch.zeros_like(fake_pred))
    disc_real_loss = adv_criterion(real_pred, torch.ones_like(real_pred))

    disc_loss = (disc_fake_loss + disc_real_loss) / 2

    return disc_loss


# def get_region_loss(condition, fakes):
#     loss = 0
#     num_images = condition.shape[0]
#     pool = torch.nn.AvgPool2d(kernel_size=4, stride=4)
#     upscale = torch.nn.Upsample(scale_factor=4, mode='nearest')
#
#     for condition, fake in zip(condition, fakes):
#         condition = condition[None, :, :, :]
#         fake = fake[None, :, :, :]
#
#         # If there was any black in the 4x4 region before, that region is all now less than 1
#         condition = upscale(pool(condition))
#         condition = torch.square(torch.square(condition))
#
#         # finds the pixel difference between the fake image and the average pixel for every 4x4 chunk
#         diff = upscale(pool(fake)) - fake
#
#         # the more black there was before, the more difference is expected/wanted
#         weighted_diff = diff * condition
#         error = torch.abs(weighted_diff)
#         loss = torch.sum(error)
#
#     loss /= num_images
#     return loss


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def initialize_model(gen, gen_opt, disc, disc_opt, device='cuda', pretrained=False, saved_model_path=None):
    if pretrained:
        loaded_state = torch.load(saved_model_path, map_location=device)
        gen.load_state_dict(loaded_state["gen"])
        gen_opt.load_state_dict(loaded_state["gen_opt"])
        disc.load_state_dict(loaded_state["disc"])
        disc_opt.load_state_dict(loaded_state["disc_opt"])
    else:
        gen = gen.apply(weights_init)
        disc = disc.apply(weights_init)


def show_tensor_images(image_tensor, num_images=16, size=(3, 256, 256)):
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
