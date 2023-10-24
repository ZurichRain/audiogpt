import os
import json


with open("/data/hypertext/sharpwang/dataset/TTS/Genshin_Dataset/paimeng/meta_data.json","r") as f:
    data = json.load(f)

# print(data[0])
print(len(data))
data_clear = set()

for d in data:
    data_clear.add(d["id"])

print(len(data))
