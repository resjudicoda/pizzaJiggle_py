from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.utils import draw_segmentation_masks
import numpy as np
import torch
import cv2 as cv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = maskrcnn_resnet50_fpn(pretrained=True, progress=False)
model = model.eval()

img = cv.imread(args["image"])
if img is None:
    sys.exit("Could not read the image.")
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
pizza_masks = output['masks']
# print(f"shape = {pizza_masks.shape}, dtype = {pizza_masks.dtype}, "
#       f"min = {pizza_masks.min()}, max = {pizza_masks.max()}")

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

# print("For the image, the following instances were detected:")
# print([inst_classes[label] for label in output['labels']])

proba_threshold = 0.5
pizza_bool_masks = output['masks'] > proba_threshold
print(f"shape = {pizza_bool_masks.shape}, dtype = {pizza_bool_masks.dtype}")

# There's an extra dimension (1) to the masks. We need to remove it
pizza_bool_masks = pizza_bool_masks.squeeze(1)

output = draw_segmentation_masks(orig, pizza_bool_masks, alpha=0.9).numpy().transpose(1, 2, 0)

cv.imshow("image", output)
cv.waitKey()