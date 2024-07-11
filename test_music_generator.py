import cv2
from Generator.MusicSynthGen import VerovioGenerator
from data_augmentation.data_augmentation import augment
import os
import random

#generator = VerovioGenerator("Data/MSCorePianoSynth/val.txt", fixed_number_systems=True, tokenization_method="bekern")
generator = VerovioGenerator("Data/GrandStaff/partitions_grandstaff/types/train.txt", fixed_number_systems=True, tokenization_method="bekern")


#Random sample an image from "Data/PolishScores/polish_corpus_v2_dataset and get the shape
image_path = "Data/Mozarteum/mozarteum_dataset"  # Path to the image dataset

# Get a random image file from the dataset
image_files = [img for img in os.listdir(image_path) if img.endswith(".jpg")]
random_image_file = random.choice(image_files)
image_file_path = os.path.join(image_path, random_image_file)
img = cv2.imread(image_file_path, 0)
print(img.shape)

image, ground_truth = generator.generate_score(num_sys_gen=6, cut_height=False, random_margins=False, 
                                               add_texture=True, include_title=False, include_author=False, reduce_ratio=0.5)

#image, ground_truth = generator.generate_system(reduce_ratio=0.5, add_texture=False)

#print(ground_truth)
#image_tensor = augment(image)

cv2.imwrite("test.png", image)

with open("test.krn", "w") as krnfile:
    krnfile.write("**kern\t**kern\n" + "".join(ground_truth[1:-1]).replace('<s>', ' ').replace('<b>', '\n').replace('<t>', '\t'))

