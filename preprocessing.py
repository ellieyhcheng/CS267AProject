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

def preprocess():
    pickle_file = 'patterns.pickle'
    count = 1268
    end = 1700

    with open(pickle_file, 'ab') as pf:
        with open(os.path.join('test_set2', 'test.csv')) as csvfile:
            with open('dataset.csv', 'a') as datafile:
                # datafile.write('artist,patternId,previewImage,palette,rating\n')

                reader = list(csv.DictReader(csvfile))
                while (count < end) and (count < len(reader)):
                    row = reader[count]
                    if count % 100 == 0:
                        print("Finished", count, "patterns")

                    img_num = row['patternId']

                    palette = row['palette'].strip().split(' ')
                    # print(img_num)
                    testimg_file = os.path.join('test_set2', str(img_num)+'.png')
                    with Image.open(testimg_file) as imgfile:
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

                    res = requests.get("http://www.colourlovers.com/api/pattern/" + str(img_num), params={"format": "json"})

                    while (res.status_code != 200):
                        print(img_num, ':', 'RETRY')
                        res = requests.get("http://www.colourlovers.com/api/pattern/" + str(img_num), params={"format": "json"})

                    pattern = res.json()[0]

                    hearts = pattern['numHearts']
                    views = pattern['numViews']
                    
                    r = (hearts - (0.0152 * views - 0.263)) / (0.0128 * views + 0.218) + 3

                    patt = Pattern(img_num, width, height, segments, px2id, enc_str, new_palette, r)
                    # print('Dumping')
                    pickle.dump(patt, pf, protocol=4)
                    metadata = row['artist'] + ',' + str(row['patternId']) + ','  + row['previewImage'] + ',' + row['palette']  + ',' + str(r) + '\n'
                    datafile.write(metadata)

        print("Finished all color groups...")
        
if __name__ == "__main__":
    preprocess()