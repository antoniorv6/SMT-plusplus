from PIL import Image
from torchvision import transforms
from .transforms_custom import *

import numpy as np

def augment(image):
    distortion_perspective = np.random.uniform(0,0.3)

    elastic_dist_magnitude = np.random.randint(1, 20 + 1)
    elastic_dist_kernel = np.random.randint(1, 3 + 1)
    magnitude_w, magnitude_h = (elastic_dist_magnitude, 1) if np.random.randint(2) == 0 else (1, elastic_dist_magnitude)

    ctr_factor = np.random.uniform(0.7, 2)

    transform = transforms.Compose(
        [   
            transforms.ToPILImage(),
            transforms.RandomApply([DPIAdjusting(np.random.uniform(0.75, 1))], p=0.1),
            transforms.RandomPerspective(distortion_scale=distortion_perspective, p=0.1, interpolation=Image.BILINEAR, fill=255),
            transforms.RandomApply([ElasticDistortion(grid=(elastic_dist_kernel, elastic_dist_kernel), magnitude=(magnitude_w, magnitude_h), min_sep=(1,1))], p=0.1),
            transforms.RandomApply([RandomTransform(16)], p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(3, 5))], p=0.1),
            transforms.RandomApply([ContrastAdjust(factor=ctr_factor)], p=0.1),
            transforms.RandomInvert(p=1.0),
            transforms.Grayscale(),
            transforms.ToTensor()
        ]
    )
    
    image = transform(image)

    return image

def convert_img_to_tensor(image):
    transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.RandomInvert(p=1.0),
        transforms.Grayscale(),
        transforms.ToTensor()]
    )

    image = transform(image)

    return image