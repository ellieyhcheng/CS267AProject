import requests
from skimage.color import rgb2lab, rgb2hsv
import csv
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
import time
import datetime
from math import cos, sin
import numpy as np
from segment import segment_image, get_color_groups, hex2rgb, enclosure_strengths
from thing import ColorGroup, Pattern, ColorGroupSegment

folder = 'good'

def preprocess():
    id2palette = {}

    with open('good_dataset.csv') as gd:
        reader = csv.DictReader(gd)
        for row in reader:
            id2palette[row['id']] = row['palette']

    pickle_file = '{}_patterns.pickle'.format(folder)

    count = 0

    with open(pickle_file, 'wb') as pf:
            with open('{}_dataset2.csv'.format(folder), 'w') as datafile:
                    images = os.listdir(folder)
                    for im in images:
                        img_num = im.split('.')[0]

                        if not img_num in id2palette:
                            res = requests.get("http://www.colourlovers.com/api/pattern/" + img_num, params={"format": "json"})

                            while res.status_code != 200:
                                res = requests.get("http://www.colourlovers.com/api/pattern/" + img_num, params={"format": "json"})
                            p = res.json()[0]
                            # print(p)
                            palette = p['colors']
                        else:
                            palette = id2palette[img_num].split()
                            # print(palette)

                        with Image.open(os.path.join(folder, im)) as imgfile:
                            img = imgfile.convert('RGBA')
                            img = np.array(img)

                        # print('Creating pattern')
                        print(count, ':', img_num)
                        count += 1

                        height,width,d = img.shape

                        # print('Segmenting')
                        start = time.time()
                        print('    Start:', datetime.datetime.fromtimestamp(start).strftime('%H:%M:%S'))
                        segments,px2id,adjacency = segment_image(img, palette)
                        print('    Adj:', len(adjacency))
                        if len(adjacency) > 5000:
                            print('    TOO MANY SEGMENTS')
                            continue
                        # print('Counting enclosure_strengths')
                        enc_str = enclosure_strengths(px2id, len(adjacency), adjacency)
                        endtime = time.time()
                        print('    End:', datetime.datetime.fromtimestamp(endtime).strftime('%H:%M:%S'))

                        if endtime - start > 240: # 240 seconds = 4 min
                            print('    OVER TIME LIMIT:', endtime - start)
                            continue

                        new_palette = []
                        for color in palette:
                            if len(segments[color]) == 0:
                                segments.pop(color, None)
                            else:
                                new_palette.append(color)

                        patt = Pattern(img_num, width, height, segments, px2id, enc_str, new_palette, 0)
                        # print('Dumping')
                        pickle.dump(patt, pf, protocol=4)
                        
                        # metadata = p["userName"] + ',' + str(p['id']) + ',' + p['imageUrl'] + ',' + ' '.join(new_palette) + '\n'
                        metadata = "{},{}\n".format(img_num, ' '.join(new_palette))
                        datafile.write(metadata)
                
if __name__ == "__main__":
    preprocess()