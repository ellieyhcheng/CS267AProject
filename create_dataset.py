import json
import requests

res = requests.get("http://www.colourlovers.com/api/patterns/top?format=json&numResults=100")

patterns = res.content

for p in patterns:
    print(p["id"])