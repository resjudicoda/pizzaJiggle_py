from pickletools import uint8
from torchvision.models import detection
from torchvision.utils import draw_segmentation_masks
import numpy as np
import torch
import cv2 as cv
import argparse
import matplotlib.pyplot as plt
import os

maskrcnn = detection.maskrcnn_resnet50_fpn
retinanet = detection.retinanet_resnet50_fpn

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = maskrcnn(pretrained=True, progress=False)
model = model.eval()

img = cv.imread(args["image"])

if img is None:
    sys.exit("Could not read the image.")
img_org = img
# convert to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
rgb_org = img
# convert to channels first
img = img.transpose((2, 0, 1))
orig = torch.ByteTensor(img)
# add the batch dimension
img = np.expand_dims(img, axis=0)
# scale tensor values to range 0-1
img = img / 255.0
# convert to floating point tensor
img = torch.FloatTensor(img)
# send the input to the device and pass the it through the network to
# get the detections and predictions
img = img.to(DEVICE)
output = model(img)[0]

inst_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


inst_class_to_idx = {cls: idx for (idx, cls) in enumerate(inst_classes)}
# find the index number of the pizza label
pizza_index = inst_class_to_idx['pizza']

pizza_mask = output['masks'][output['labels'] == pizza_index]

proba_threshold = 0.5
score_threshold = .75

# select for masks that meet the proba threshold and match the pizza_index
pizza_bool_masks = pizza_mask > proba_threshold

img_mask = pizza_bool_masks.squeeze(1).detach().numpy().transpose(1, 2, 0)

# There's an extra dimension (1) to the masks. We need to remove it
pizza_bool_masks = pizza_bool_masks.squeeze(1)
# this draws the mask on the original image
masked_output = draw_segmentation_masks(orig, pizza_bool_masks, alpha=0.7).numpy().transpose(1, 2, 0)

# cropping, from stack overflow (https://stackoverflow.com/questions/40824245/how-to-crop-image-based-on-binary-mask)

#change mask to float - white pizza shape with black background
float_mask = np.float32(img_mask)
#create float of original image
img_255 = img_org / 255
image_mat = np.float32(img_255)

# #change mask to a 3 channel image - looks similar to float_mask
src1_mask=cv.cvtColor(float_mask,cv.COLOR_GRAY2BGR)
#subtract mask from image - yields only slice, but in blue tone 
mask_out=cv.subtract(src1_mask, image_mat)
# returns image to original without crop
final_mask_out=cv.subtract(src1_mask, mask_out)

# multiplying the image by the mask works to crop the image
x_mask_out = image_mat * src1_mask
print("x_mask_out", type(x_mask_out), x_mask_out.ndim, x_mask_out.shape)

# directory = r'/Users/adamreis/Coding/pizzaJiggle_py/images'
# os.chdir(directory)
# filename = 'savedImage.png'
# cv.imwrite(filename, x_mask_out)

cv.imshow("image", x_mask_out)
cv.waitKey()