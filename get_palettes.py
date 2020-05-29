import json
import requests
import urllib.request
# from skimage import io, color
f = open("compat_set.csv", 'a')
# f.write('artist,paletteId,previewImage,rating,palette\n')

for i in range(11,12): #, 'orderCol': "numVotes", 'sortBy': 'DESC'
    res = requests.get("https://www.colourlovers.com/api/palettes", params={"format": "json", "numResults": 100, 'resultOffset': i * 100, 'orderCol': "numViews", 'sortBy': 'ASC'})

    palettes = res.json()

    for p in palettes:

        if len(p['colors']) == 0:
            continue
        hearts = p['numVotes']
        views = p['numViews']
        
        r = (hearts - (0.0152 * views - 0.263)) / (0.0128 * views + 0.218) + 3
        if r > 5:
            r = 5
        print(p["id"], r)

        # f.write(p["userName"] + ',' + str(p['id']) + ',' + p['url'] + ',' + str(r) + ',')

        # for c in p["colors"]:
        #     f.write(str(c) + " ")
        # f.write('\n')

f.close()