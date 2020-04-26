import json
import requests
import urllib.request
# from skimage import io, color

res = requests.get("http://www.colourlovers.com/api/patterns/top", params={"format": "json"})

patterns = res.json()

for p in patterns:
    print(p["id"])
    urllib.request.urlretrieve(p["imageUrl"], str(p["id"]) + ".png")

    with open(str(p["id"]) + ".txt", 'w') as f:        
        for c in p["colors"]:
            f.write(str(c) + "\n")

# lab = color.rgb2lab(rgb)
# map each pixel to a the closest color in source palette
