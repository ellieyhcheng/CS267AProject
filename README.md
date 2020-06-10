# CS267A Project: Graphic Generator

Graphic generator from an input image using k-means, probabilistic factor graph, and MCMC sampling. 

## How to generate an image
```Python
python3 main.py images/1879321.png --iter 200 --num 3 --clusters 5
```
Resulting images will be in `results/`. At least 100 sampling iterations is necessary to generate somewhat decent graphics. 500 iterations is recommended. 

## Notes
Constraints:
- If the input image's width or height is greater than 320px, the generated image is resized to max of 200px in width or height.
- Number of segments, including noise, in the image cannot be greater than 5000. (Resizing should take care of this already)
- O'Donovan compatibility works for up to 5 colors in a palette, so number of clusters cannot be greater than 5.
