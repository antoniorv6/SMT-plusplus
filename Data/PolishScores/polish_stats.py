#Make a python program that extracts the largest image in a folder (including subfolders) and prints its size and path.

import os
from PIL import Image

def get_largest_image(path):
    largest_image = None
    largest_size = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                size = image.size[0] * image.size[1]
                if size > largest_size:
                    largest_size = size
                    largest_image = image_path
    return largest_image, largest_size

path = "polish_corpus_v0_dataset"
largest_image, largest_size = get_largest_image(path)
print(f"Largest image: {largest_image}")
print(f"Size: {largest_size} pixels")
# Print height and width
image = Image.open(largest_image)
print(f"Height: {image.size[1] * 0.4} pixels")
print(f"Width: {image.size[0] * 0.4} pixels")

