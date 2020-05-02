from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt
from skimage import color

def lab_dist(rgb1, rgb2):
    lab1 = color.rgb2lab([[rgb1]])[0][0]
    lab2 = color.rgb2lab([[rgb2]])[0][0]
    return (lab1[0]-lab2[0])**2+(lab1[1]-lab2[1])**2+(lab1[2]-lab2[2])**2

def rgb_dist(rgb1, rgb2):
    return (rgb1[0]-rgb2[0])**2+(rgb1[1]-rgb2[1])**2+(rgb1[2]-rgb2[2])**2

def hex_dist(hex1, hex2):
    return rgb_dist(hex2rgb(hex1),hex2rgb(hex2))

def get_nearest_hex(rgb_hex,palette):
    #returns nearest color in hex
    dist = [hex_dist(rgb_hex, p_hex) for p_hex in palette]
    return palette[dist.index(min(dist))]

def get_nearest_rgb(rgb, palette):
    return get_nearest_hex(rgb2hex((rgb[0],rgb[1],rgb[2])), palette)

def rgb2hex(rgb):
    return ('%02x%02x%02x' % rgb).upper()

def hex2rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))

def getsegment(img,i,j,palette,visited):
    #do bfs to get image segment
    h,w,d = img.shape
    col = get_nearest_rgb(img[i][j],palette)

    segment = []
    q = []
    q.append((i,j))
    visited.add((i,j))
    while len(q) > 0:
        pi,pj = q.pop(0)
        segment.append((pi,pj))
        
        if pi > 0 and (pi-1,pj) not in visited and get_nearest_rgb(img[pi-1][pj],palette) == col:
            q.append((pi-1,pj))
            visited.add((pi-1,pj))
        if pi+1 < h and (pi+1,pj) not in visited and get_nearest_rgb(img[pi+1][pj],palette) == col:
            q.append((pi+1,pj))
            visited.add((pi+1,pj))
        if pj > 0 and (pi,pj-1) not in visited and get_nearest_rgb(img[pi][pj-1],palette) == col:
            q.append((pi,pj-1))
            visited.add((pi,pj-1))
        if pj+1 < w and (pi,pj+1) not in visited and get_nearest_rgb(img[pi][pj+1],palette) == col:
            q.append((pi,pj+1))
            visited.add((pi,pj+1))
    return segment

def segment_image(img, palette):
    img_cpy = img.copy()

    segments = {}
    for col in palette:
        segments[col] = []

    visited = set()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (i,j) in visited:
                continue
            rgb_hex = get_nearest_rgb(img_cpy[i][j],palette)
            seg = getsegment(img_cpy,i,j,palette,visited)
            segments[rgb_hex].append(seg)

    return segments
    

def test(img_num):
    testimg_file = 'test_set2/'+str(img_num)+'.png'

    testimg = Image.open(testimg_file)
    testimg = testimg.convert('RGBA')
    testimg = np.array(testimg)
    print(testimg.shape)

    # testimg = np.array(Image.open(testimg_file))
    # print(testimg.shape)

    palette = ['F5F1E9', 'D12A2A', '9C1313', '570606', '783535'] #1061 test2
    #palette = ['87712F', '45201E', 'B8A879', '5D857D', 'FF0F53'] #92013 test2
    #palette = ['4F0037', '63386D', '687899', 'D1E6D3', 'ECEDC5'] # test1
    img_segmented = segment_image(testimg, palette)
    print(len(img_segmented))

    total_segs = 0
    for col in palette:
        total_segs += len(img_segmented[col])
    print(total_segs)

    for segment in img_segmented['F5F1E9']:
        for px in segment:
            testimg[px[0]][px[1]] = [255,255,255,255]
    
    plt.imshow(testimg)
    plt.show()


if __name__ == '__main__':
    test(1061)