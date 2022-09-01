import cv2 as cv
import sys
import numpy as np
import argparse

MEDIAN_KERNAL_SIZE = 7
BILATERAL_KERNAL_SIZE = 12
MAX_KERNEL_LENGTH = 31

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

# img = cv.imread(args["image"])
# cv.imshow("image", img)

# if img is None:
#     sys.exit("Could not read the image.")

def cartoonify(img):

    # step 1: blur and detect edges
    # apply median blur then canny edge detection

    # for i in range(1, MEDIAN_KERNAL_SIZE, 2):
    #     median = cv.medianBlur(img, i)

    # median_edges = cv.Canny(median, 100, 200)

    # step 2: downsize image, bilateral filter, upsize, blur for smoothing
    down_scale_percent = 25  # percent of original size
    downsize_width = int(img.shape[1] * down_scale_percent / 100)
    downsize_height = int(img.shape[0] * down_scale_percent / 100)
    downsize_dim = (downsize_width, downsize_height)

    # downsize image
    downsized = cv.resize(img, downsize_dim, interpolation=cv.INTER_AREA)

    # apply bilteral filter to downsized image
    bilateral = downsized

    for j in range(0, 14):
        for i in range(1, BILATERAL_KERNAL_SIZE, 2):
            bilateral = cv.bilateralFilter(bilateral, i, i * 2, i / 2)

    upsize_width = int(img.shape[1])
    upsize_height = int(img.shape[0])
    upsize_dim = (upsize_width, upsize_height)

    # upsize the filtered image back to original size
    upsized = cv.resize(bilateral, upsize_dim, interpolation=cv.INTER_AREA)

    # apply median blur to smooth errrors of upsizing
    for i in range(1, MEDIAN_KERNAL_SIZE, 2):
        final_toon = cv.medianBlur(upsized, i)

    # combine edges and toon
    # canny produces a b&w image with no channel value so addWeighted will not work
    # median_edges_downsized = cv.resize(
    #     median_edges, downsize_dim, interpolation=cv.INTER_AREA)
    # median_edges_upsized = cv.resize(
    #     median_edges_downsized, upsize_dim, interpolation=cv.INTER_AREA)
    return final_toon

# final_toon = cartoonify(img)
# cv.imshow('Median/Upsized', final_toon)
# # Wait until user press some key
# cv.waitKey()
