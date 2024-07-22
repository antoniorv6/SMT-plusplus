import cv2
from Generator.SynthGenerator import VerovioGenerator

generator = VerovioGenerator(sources=['Data/GrandStaff/partitions_grandstaff/types/train.txt'], base_folder='Data/GrandStaff/', tokenization_mode='bekern')

image, ground_truth = generator.generate_full_page_score(max_systems=2, strict_systems=True, strict_height=False, reduce_ratio=0.5)

print(ground_truth)

cv2.imwrite('test.png', image)
with open('test.krn', 'w') as file:
    file.write("**kern\t**kern\n" + "".join(ground_truth[1:-1]).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t').replace('@', '').replace('·', ''))
