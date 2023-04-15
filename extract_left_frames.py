import cv2
import glob
import os

train_vol_names = glob.glob(os.path.join('path_to_target_folder', '*.png'))
train_vol_names.sort()
for image in train_vol_names:
  vol1 = cv2.imread(image, 1)
  vol1 = vol1[0:1024, :, :]
  cv2.imwrite(image, vol1)
 

