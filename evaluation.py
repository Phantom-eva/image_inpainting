import torch
import numpy as np
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize


def evaluate(model, dataset, device, filename, real_file, fake_file):
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    #mask = 1 - torch.tensor([m.detach().numpy() for m in mask])

    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    save_image(unnormalize(gt), real_file)
    save_image(unnormalize(output_comp), fake_file)

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)

def evaluate_test(model, dataset, device, real_file, fake_file, mask_file):
    for i in range(25):
        image, mask, gt = zip(*[dataset[j] for j in range(i*8,(i+1)*8)])
        #mask = 1 - torch.tensor([m.detach().numpy() for m in mask])

        image = torch.stack(image)
        mask = torch.stack(mask)
        gt = torch.stack(gt)

        with torch.no_grad():
            output, _ = model(image.to(device), mask.to(device))
        output = output.to(torch.device('cpu'))
        output_comp = mask * image + (1 - mask) * output

        save_image(unnormalize(image), mask_file+'img_{:d}.jpg'.format(i + 1))
        save_image(unnormalize(gt), real_file+'/real_{:d}.jpg'.format(i + 1))
        save_image(unnormalize(output_comp), fake_file+'/fake_{:d}.jpg'.format(i + 1))

