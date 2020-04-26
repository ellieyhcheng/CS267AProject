import json
import requests

res = requests.get("http://www.colourlovers.com/api/patterns/top")

print(res.content)