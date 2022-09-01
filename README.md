# pizzaJiggle_py

This is version two of one of my personal projects. It uses PyTorch/OpenCV to detect pizza and return a mask, apply a "cartoonify" effect to the image, and use the mask to crop the pizza out of the image. It then uses imageio to create a gif to simulate a jiggling animation.

Special thanks to the following article for providing a baseline for me to work out my own cartoonify feature.

https://stacks.stanford.edu/file/druid:yt916dh6570/Dade_Toonify.pdf

##Note: Currently runs in an Anaconda environment via the following shell command:

"python pizza_jiggle.py -i ./images/slice.jpg"
