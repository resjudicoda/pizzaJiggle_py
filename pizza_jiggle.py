import numpy as np
import cv2 as cv
import argparse
import os

from object_detect.object_detect import *
from cartoonify.cartoonify import cartoonify

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

# change black background to white
# test[np.where((test==[0,0,0]).all(axis=2))] = [255,255,255]

# delete black background
# test = delete_black_background(cropped)

# jiggle

# display it
cv.imshow("image", cropped)
cv.waitKey()