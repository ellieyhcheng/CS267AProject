import numpy as np
import cv2

from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, rgb2hsv, lab2rgb
import os

folder = 'landscape'

def cluster(img_num):
    testimg = Image.open(os.path.join(folder,'{}.png'.format(img_num)))
    testimg = testimg.convert('RGB')
    ori_w, ori_h = testimg.size
    if ori_h or ori_h > 320:
        resize_ratio = min(200/ori_w, 200/ori_h)
        newsize = (int(resize_ratio*ori_w),int(resize_ratio*ori_h) )
        img = testimg.resize(newsize) 
    img = np.array(img)

    vectorized = np.float32(img.reshape((-1,3)))

    k = 5
    attempts = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center = cv2.kmeans(vectorized,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)
    print(center)

    res = center[label.flatten()]
    # print(res)
    result_image = res.reshape((img.shape))

    fig, ax = plt.subplots(2,4, figsize=(9,2))
    fig.suptitle(img_num)
    ax[0][0].imshow(img)

    ax[0][1].imshow(np.where(label == 0, center[0], [255, 255, 255]).reshape(img.shape))
    ax[0][2].imshow(np.where(label == 1, center[1], [255, 255, 255]).reshape(img.shape))
    ax[0][3].imshow(np.where(label == 2, center[2], [255, 255, 255]).reshape(img.shape))
    ax[1][0].imshow(np.where(label == 3, center[3], [255, 255, 255]).reshape(img.shape))
    ax[1][1].imshow(np.where(label == 4, center[4], [255, 255, 255]).reshape(img.shape))
    ax[1][2].imshow(result_image)
    for i in range(0,2):
        for j in range(0,4):
            ax[i][j].axis('off')
    fig.delaxes(ax[1][3])
    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    # for i in os.listdir(folder)[:1]:
    #     cluster(i.split('.')[0])
    cluster(30)