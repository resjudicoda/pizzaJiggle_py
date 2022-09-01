from jiggle.jiggle import vertical_waves
import numpy as np
import cv2 as cv
import argparse
import os
import imageio

from object_detect_and_crop.object_detect_and_crop import *
from cartoonify.cartoonify import cartoonify
from jiggle.jiggle import *

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

img = cv.imread(args["image"])

if img is None:
    sys.exit("Could not read the image.")

# detect pizza and get mask
mask = object_detect(img)

# cartoonify
tooned = cartoonify(img)

# crop tooned image
cropped = crop(tooned, mask)

# optional change black background to white
# cropped[np.where((cropped==[0,0,0]).all(axis=2))] = [255,255,255]

# optional delete black background
# test = delete_black_background(cropped)

# jiggle
# outputs a gif
# create a version of the cropped image with vertical waves
wavy = vertical_waves(cropped)

# convert bgr to rgb for imageio
cropped_rgb = cv.cvtColor(cropped, cv.COLOR_BGR2RGB)
wavy_rgb = cv.cvtColor(wavy, cv.COLOR_BGR2RGB)

# convert images from float32 to uint8 for imageio
def normalize8(I):
  mn = I.min()
  mx = I.max()

  mx -= mn

  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)

cropped_rgb_uint8 = normalize8(cropped_rgb)
wavy_rgb_uint8 = normalize8(wavy_rgb)

# append regular and wavy image to list
image_list = []
image_list.append(cropped_rgb_uint8)
image_list.append(wavy_rgb_uint8)
# create gif using images list
imageio.mimwrite('animated_from_images.gif', image_list)


# display image if needed
# cv.imshow("image", wavy)
# cv.waitKey()