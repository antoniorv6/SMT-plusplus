import cv2
from Generator.MusicSynthGen import VerovioGenerator
from data_augmentation.data_augmentation import convert_img_to_tensor
from skimage.filters import threshold_sauvola
#Utilizar otsu thresholding para binarizar la imagen

#generator = VerovioGenerator("Data/GrandStaff/partitions_grandstaff/types/train.txt", fixed_number_systems=True)
#
#image, ground_truth = generator.generate_score(num_sys_gen=6, cut_height=False, random_margins=False,
#                                               add_texture=True, include_title=False, include_author=False)
# Apply sauvola binarization to the image
image = cv2.imread("11.jpg")
image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
thresh_sauvola = threshold_sauvola(image, window_size=25)
image = image < thresh_sauvola
mask = image.astype(float)
# Dilate ones
#mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=7)
# Save mask
cv2.imwrite("mask.png", mask * 255)
