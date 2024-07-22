import os
import cv2

sizes = []
for sample in os.listdir('Data/Mozarteum/mozarteum_dataset'):
    if sample.endswith('.jpg'):
        img = cv2.imread(f'Data/Mozarteum/mozarteum_dataset/{sample}', 0)
        sizes.append(img.shape)

#Print maximum size, minimum size and avg size of the shapes stored in sizes. Bear in mind that the content of sizes are tuples
max_size = max(sizes)
min_size = min(sizes)
print(f"Maximum size: {max_size}")
print(f"Minimum size: {min_size}")
