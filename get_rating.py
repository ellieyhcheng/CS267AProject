import json
import requests
import urllib.request
from thing import Pattern, ColorGroup, ColorGroupSegment
import pickle

f = open("test_set2/test.csv", 'r')
h = open("ratings.pickle", 'ab')
ratings = []
all_patterns = []
with open('patterns.pickle', 'rb') as pf:
    while 1:
        try:
            all_patterns.append(pickle.load(pf))
        except EOFError:
            break

count = 0
for patt in all_patterns:
    res = requests.get("http://www.colourlovers.com/api/pattern/" + str(patt.img_num), params={"format": "json"})
    
    while (res.status_code != 200):
        print(patt.img_num, ':', 'RETRY')
        res = requests.get("http://www.colourlovers.com/api/pattern/" + str(patt.img_num), params={"format": "json"})

    pattern = res.json()[0]

    hearts = pattern['numHearts']
    views = pattern['numViews']
    
    r = (hearts - (0.0152 * views - 0.263)) / (0.0128 * views + 0.218) + 3
    print(count, ':', r)
    count += 1
    ratings.append(r)

pickle.dump(ratings, h, protocol=4)

f.close()
h.close()
# lab = color.rgb2lab(rgb)
# map each pixel to a the closest color in source palette
