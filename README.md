# pizzaJiggle_py

This is version two of one of my personal projects. As of now, it consists of two Python scripts: one that uses PyTorch/OpenCV to detect pizza and crop it out of the image; and one that uses OpenCV to apply a "cartoonify" effect to an image.

I am working on a third script to animate the output image, as well as the main script to connect everything together.

Special thanks to the following article for providing a baseline for me to work out my own cartoonify feature.

https://stacks.stanford.edu/file/druid:yt916dh6570/Dade_Toonify.pdf

##Note: The scripts currently run via the following shell commands:

Cartoonify: "python cartoonify/cartoonify.py -i ./images/slice.jpg"
Object Detection: "python object_detect/object_detect.py -i ./images/slice.jpg"

python pizza_jiggle.py -i ./images/slice.jpg
