import cv2
from Generator.MusicSynthGen import VerovioGenerator
from data_augmentation.data_augmentation import augment

generator = VerovioGenerator("Data/GrandStaff/partitions_grandstaff/types/train.txt", fixed_number_systems=True, tokenization_method="bekern")

image, ground_truth = generator.generate_score(num_sys_gen=7, cut_height=False, random_margins=False, 
                                               add_texture=True, include_title=False, include_author=False,
                                               check_generated_systems=True, page_size=[2980, 3728], reduce_ratio=0.35)

print(ground_truth)
#image_tensor = augment(image)

cv2.imwrite("test.png", image)

with open("test.krn", "w") as krnfile:
    krnfile.write("**kern\t**kern\n" + "".join(ground_truth[1:-1]).replace('<s>', ' ').replace('<b>', '\n').replace('<t>', '\t'))

