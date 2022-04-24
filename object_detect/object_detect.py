from torchvision.models import detection
import numpy as np
import torch
import cv2 as cv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

img = cv.imread(args["image"])
cv.imshow("image", img)

if img is None:
    sys.exit("Could not read the image.")



model = detection.retinanet_resnet50_fpn(pretrained=True)
model.eval()
