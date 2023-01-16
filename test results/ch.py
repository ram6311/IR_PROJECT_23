import requests
res = requests.post('http://34.72.134.105:8080/get_pagerank', json=[1,5,8])
data = res.json()
print(data)

import requests
res = requests.post('http://34.72.134.105:8080/get_pageview', json=[7199,8615,14076])
data = res.json()
print(data)