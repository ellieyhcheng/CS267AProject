from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt
# from skimage import color
import os

def rgb_dist(rgb1, rgb2):
    return (int(rgb1[0])-int(rgb2[0]))**2+(int(rgb1[1])-int(rgb2[1]))**2+(int(rgb1[2])-int(rgb2[2]))**2

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

def get_color(img,i,j,palette):
    #gets hex value in palette of nearest color in 8 neighboring pixels
    h,w,d = img.shape
    c1 = img[i][j]
    c1hex = rgb2hex((c1[0],c1[1],c1[2]))
    if c1hex in palette:
        return c1hex
    colors = []
    distances = []

    for a in range(-1,2):
        for b in range(-1,2):
            if (a == 0 and b == 0) or i+a < 0 or i+a >= h or j+b < 0 or j+b >= w :
                continue
            c2 = img[i+a][j+b]
            # if rgb2hex((c2[0],c2[1],c2[2])) not in palette:
            #     continue
            colors.append(c2)
            d = rgb_dist(c1,c2)
            distances.append(d)
    # if len(distances) == 0:
    #     return get_nearest_hex(c1hex,palette)
    return get_nearest_rgb(colors[distances.index(min(distances))],palette)
    #return rgb2hex((n_rgb[0],n_rgb[1],n_rgb[2]))
    
    # return get_nearest_hex(rgb2hex((n_rgb[0],n_rgb[1],n_rgb[2])),palette)

def getsegment(img,i,j,palette,visited,px2id,adjacency,seg_id):
    #do bfs to get image segment
    h,w,d = img.shape
    # col = get_nearest_rgb(img[i][j],palette)
    col = get_color(img,i,j,palette)

    segment = set()
    q = []
    q.append((i,j))
    visited.add((i,j))
    while len(q) > 0:
        pi,pj = q.pop(0)
        segment.add((pi,pj))
        px2id[pi][pj] = seg_id

        # if pi > 0 and (pi-1,pj) not in visited and get_nearest_rgb(img[pi-1][pj],palette) == col:
        if pi > 0:                
            if (pi-1,pj) not in visited and get_color(img,pi-1,pj,palette) == col:
                q.append((pi-1,pj))
                visited.add((pi-1,pj))
            elif px2id[pi-1][pj] != -1 and px2id[pi-1][pj] != seg_id:
                adj_id = px2id[pi-1][pj]
                if seg_id not in adjacency:
                    adjacency[seg_id] = set()
                if adj_id not in adjacency:
                    adjacency[adj_id] = set()
                adjacency[seg_id].add(adj_id)
                adjacency[adj_id].add(seg_id)
        # if pi+1 < h and (pi+1,pj) not in visited and get_nearest_rgb(img[pi+1][pj],palette) == col:
        if pi+1 < h:
            if (pi+1,pj) not in visited and get_color(img,pi+1,pj,palette) == col:
                q.append((pi+1,pj))
                visited.add((pi+1,pj))
            elif px2id[pi+1][pj] != -1 and px2id[pi+1][pj] != seg_id:
                adj_id = px2id[pi+1][pj]
                if seg_id not in adjacency:
                    adjacency[seg_id] = set()
                if adj_id not in adjacency:
                    adjacency[adj_id] = set()
                adjacency[seg_id].add(adj_id)
                adjacency[adj_id].add(seg_id)
        # if pj > 0 and (pi,pj-1) not in visited and get_nearest_rgb(img[pi][pj-1],palette) == col:
        if pj > 0:
            if (pi,pj-1) not in visited and get_color(img,pi,pj-1,palette) == col:
                q.append((pi,pj-1))
                visited.add((pi,pj-1))
            elif px2id[pi][pj-1] != -1 and px2id[pi][pj-1] != seg_id:
                adj_id = px2id[pi][pj-1]
                if seg_id not in adjacency:
                    adjacency[seg_id] = set()
                if adj_id not in adjacency:
                    adjacency[adj_id] = set()
                adjacency[seg_id].add(adj_id)
                adjacency[adj_id].add(seg_id)
        # if pj+1 < w and (pi,pj+1) not in visited and get_nearest_rgb(img[pi][pj+1],palette) == col:
        if pj+1 < w:
            if (pi,pj+1) not in visited and get_color(img,pi,pj+1,palette) == col:
                q.append((pi,pj+1))
                visited.add((pi,pj+1))
            elif px2id[pi][pj+1] != -1 and px2id[pi][pj+1] != seg_id:
                adj_id = px2id[pi][pj+1]
                if seg_id not in adjacency:
                    adjacency[seg_id] = set()
                if adj_id not in adjacency:
                    adjacency[adj_id] = set()
                adjacency[seg_id].add(adj_id)
                adjacency[adj_id].add(seg_id)
    return list(segment)

# matrix = pixel to segment_id
def enclosure_strengths(matrix, num_ids, adjacency):
    n = len(matrix)
    m = len(matrix[0])
    dist = 2
    count = [[0 for i in range(num_ids+1)] for j in range(num_ids)] # row segment, col is neighboring segment, value is unnnormalized strength
    for i in range(-dist, n+dist):
        for j in range(-dist, m+dist):
            s = set()
            outofbounds = set()
            for dx in range(-dist, dist+1):
                for dy in range(-dist, dist+1):
                    nx, ny = i+dx, j+dy
                    if nx >= 0 and nx < n and ny >= 0 and ny < m:
                        if i >= 0 and i < n and j >= 0 and j < m:
                            if matrix[i][j] != matrix[nx][ny]:
                                s.add(matrix[nx][ny])
                        else:
                            outofbounds.add(matrix[nx][ny])
            for k in s:
                count[k][matrix[i][j]]+=1
            for k in outofbounds:
                count[k][num_ids] += 1

    for i in range(len(count)):
        for j in range(len(count[0])):
            if count[i][j] != 0:
                if not(j in adjacency[i]):
                    count[i][j] = 0

    # normalize
    for i in range(len(count)):
        total = sum(count[i])
        if total == 0:
            continue
        for j in range(len(count[0])):
            count[i][j] /= total
        
        count[i].pop(-1)

    total_total = sum([sum(x) for x in count])
    for i in range(len(count)):
        for j in range(len(count[0])):
            count[i][j] /= total_total
    
    return count

def segment_image(img, palette):
    img_cpy = img.copy()
    h,w,d = img.shape

    segments = {}
    for col in palette:
        segments[col] = []
    
    px2id = [[-1 for j in range(w)] for i in range(h)]
    adjacency = {}
    
    visited = set()
    seg_id = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (i,j) in visited:
                continue
            rgb_hex = get_color(img_cpy,i,j,palette)
            #rgb_hex = get_nearest_rgb(img_cpy[i][j],palette)
            seg = getsegment(img_cpy,i,j,palette,visited,px2id,adjacency,seg_id)
            segments[rgb_hex].append(seg)
            seg_id += 1

    return (segments, px2id, adjacency)
    
def preprocess_image(img_num):
    testimg_file = os.path.join('test_set2', str(img_num)+'.png')
    testimg = Image.open(testimg_file)
    testimg = testimg.convert('RGBA')
    testimg = np.array(testimg)

    with open(os.path.join('test_set2', 'test.csv')) as file:
        reader = csv.DictReader(file)
        palette = None
        for row in reader:
            if row['patternId'] == str(img_num):
                palette = row['palette'].strip().split(' ')
    
        if palette is None:
            print("Bad image ID")
            exit(2)
    return testimg, palette

def get_color_groups(img_num):
    img, palette = preprocess_image(img_num)
    segments, px2id, adjacency = segment_image(img, palette)
    color_groups = {}
    # print(adjacency)
    # print(palette)
    # img2 = img.copy()

    for color in segments:
        for seg in segments[color]:
            if (15,43) in seg:
                print("HELLO!", color)
                print(seg)
                break

    print(get_color(img2,15,43,palette))
    print(get_color(img2,150,170,palette))
    # print(get_color(img2,3,28,palette))
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if px2id[i][j] != 22:# and px2id[i][j] != 36:
                img2[i][j] = [0,0,0,255]
            else:
                print(i,j)
    plt.imshow(img2)
    plt.show()
    # print(get_color(img2,1,23,palette))
    # print(get_color(img2,1,24,palette))
    # print(get_color(img2,3,28,palette))
    # for i in range(img2.shape[0]):
    #     for j in range(img2.shape[1]):
    #         if px2id[i][j] != 36:# and px2id[i][j] != 36:
    #             img2[i][j] = [0,0,0,255]
    #         else:
    #             print(i,j)
    # plt.imshow(img2)
    # plt.show()
            
    # print(px2id)
    # print(len(adjacency))
    enc_str = enclosure_strengths(px2id, len(adjacency), adjacency)
    for id1 in adjacency:
        for id2 in adjacency[id1]:
            #id1 is adjacent to id2
            assert(enc_str[id1][id2] > 0)
    # adjacency maps from segment id to a set of all segment ids that are adjacent to it
    total = sum([sum(x) for x in enc_str])
    assert(np.round(total, 2) == 1)

    enc = enc_str[3]
    enc_map = {}
    for i in range(len(enc)):
        if enc[i] != 0:
            enc_map[i] = enc[i]
    print(enc_map)
    print(adjacency[3])

    for color in palette:
        group = np.full(img.shape, 255)
        r,g,b = hex2rgb(color)
        for segment in segments[color]:
            for px in segment:
                group[px[0]][px[1]] = [r,g,b,255]
        color_groups[color] = group
    return color_groups

def test(img_num):        
    color_groups = get_color_groups(img_num)
    # for color_group in color_groups.values():
    #     plt.imshow(color_group)
    #     plt.show()

if __name__ == '__main__':
    # test(584317)
    test(707090)
    # test(465753)
