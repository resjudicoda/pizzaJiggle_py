from torchvision.models import detection
from torchvision.utils import draw_segmentation_masks
import numpy as np
import torch
import cv2 as cv
import argparse
import matplotlib.pyplot as plt

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
#print(f"inst_class_to_idx: {inst_class_to_idx}")
# find the index number of the pizza label
pizza_index = inst_class_to_idx['pizza']

# print("For the image, the following instances were detected:")
# print([inst_classes[label] for label in output['labels']])
# print("The labels are as follows:")
# print(output['labels'])
# print("The scores are as follows:")
# print(output['scores'])
# print(f"The pizza index is: {pizza_index}")

pizza_mask = output['masks'][output['labels'] == pizza_index]
# print(f"shape = {pizza_masks.shape}, dtype = {pizza_masks.dtype}, "
#       f"min = {pizza_masks.min()}, max = {pizza_masks.max()}")

proba_threshold = 0.5
score_threshold = .75

#pizza_bool_masks = output['masks'][output['scores'] > score_threshold]  > proba_threshold

# select for masks that meet the proba threshold and match the pizza_index
pizza_bool_masks = pizza_mask > proba_threshold
#print(f"shape = {pizza_bool_masks.shape}, dtype = {pizza_bool_masks.dtype}")

# There's an extra dimension (1) to the masks. We need to remove it
pizza_bool_masks = pizza_bool_masks.squeeze(1)
# trying to use the mask to crop the image
img_mask = pizza_mask.squeeze(1).detach().numpy().transpose(1, 2, 0)

output = draw_segmentation_masks(orig, pizza_bool_masks, alpha=0.7).numpy().transpose(1, 2, 0)

#test = pizza_bool_masks.numpy().transpose(1, 2, 0)

#cv.imshow("test", test)
# cv.imshow("image", output)
# cv.waitKey()

# for image cropping, not working
for h in range(len(img_mask)):
    for w in range(len(img_mask)):
        if img_mask[h][w][0] == 0:
            for i in range(3):
                img_org[h][w][i] = 0
        else:
            continue
#this wasnt working either
#masked = cv.bitwise_and(img_org, img_org, img_mask)
cv.imshow("image", img_org)
cv.waitKey()       
# plt.imshow(img_org)