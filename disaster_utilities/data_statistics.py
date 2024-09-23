import os
import cv2
import numpy as np

# Function that calculates the area of an image
def calculate_area(img):
    return img.shape[0] * img.shape[1]

sizes = []
images = []
for sample in os.listdir('Data/Polish_Scores/normalized_images'):
    if sample.endswith('.jpg'):
        img = cv2.imread(f'Data/Polish_Scores/normalized_images/{sample}', 0)
        sizes.append(calculate_area(img))
        images.append(img)

#Print maximum size, minimum size and avg size of the shapes stored in sizes. Bear in mind that the content of sizes are tuples
max_size = max(sizes)
min_size = min(sizes)
cv2.imwrite('lowest_image.jpg', images[np.argmin(sizes)])
print(f"Maximum size: {images[np.argmax(sizes)].shape}")
print(f"Minimum size: {images[np.argmin(sizes)].shape}")
