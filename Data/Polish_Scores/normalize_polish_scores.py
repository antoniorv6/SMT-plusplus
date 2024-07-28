import os
import cv2
import numpy as np

os.makedirs('normalized_images', exist_ok=True)

normal_shape = (2970, 2100)

for file in os.listdir('polish_scores_dataset'):
    if file.endswith('.jpg'):
        img = cv2.imread(f'polish_scores_dataset/{file}', 0)
        img = cv2.resize(img, normal_shape)
        #Check if the shape is above the limit
        reduction_x = 2970 / img.shape[0]
        reduction_y = 2100 / img.shape[1]
        reduction = min(reduction_x, reduction_y)
        img = cv2.resize(img, (int(img.shape[0]*reduction), int(img.shape[1]*reduction)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'polish_scores_dataset/{file}', img)