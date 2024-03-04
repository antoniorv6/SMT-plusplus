import cv2
from Generator.MusicSynthGen import VerovioGenerator

generator = VerovioGenerator("Data/GrandStaff/partitions_grandstaff/types/train.txt", fixed_number_systems=True)

image, ground_truth = generator.generate_score(num_sys_gen=2, cut_height=False, add_texture=False, include_title=False, include_author=False)
cv2.imwrite("test.png", image)

with open("test.krn", "w") as krnfile:
    krnfile.write("**kern\t**kern\n" + "".join(ground_truth[1:-1]).replace('<s>', ' ').replace('<b>', '\n').replace('<t>', '\t'))

