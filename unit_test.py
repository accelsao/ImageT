import numpy as np
import torch
from torchvision import transforms

if __name__ == '__main__':

    image_size = 256
    transform_func = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size, image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    input = torch.randn(256, 256, 3).numpy().astype(np.uint8)
    print(input.shape)
    out = transform_func(input)
    print(out)