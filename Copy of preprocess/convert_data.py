import os
import json

# ori_data = []
# with open("/data/hypertext/sharpwang/dataset/TTS/leidain/leidian.cleaned","r") as f:
#     for line in f.readlines():
#         ori_data.append(line.strip().split("|"))

# new_data = []
# for line in ori_data:
#     new_data.append({
#         "id":line[0],
#         "qalis": [[line[2],line[2]]],
#         "wavid": line[0]
#     })

# with open("/data/hypertext/sharpwang/dataset/TTS/leidain/meta_data.json","w") as f:
#     f.write(
#         json.dumps(new_data,ensure_ascii=False)
#     )


# with open("/data/hypertext/sharpwang/dataset/TTS/leidain/meta_data.json","r") as f:
#     data = json.load(f)
# print(data[0])

import copy

with open("/data/hypertext/sharpwang/dataset/TTS/Genshin_Dataset/paimeng/meta_data.json","r") as f:
    data = json.load(f)
    
new_data = []
for d in data:
    
    new_d = copy.deepcopy(d)
    new_d["qalis"][0][1] = d["qalis"][0][1].replace("："," -> ")
    # new_d[]
    # print(new_d)
    # exit()
    new_data.append(new_d)

with open("/data/hypertext/sharpwang/dataset/TTS/Genshin_Dataset/paimeng/meta_1_data.json","w") as f:
    f.write(json.dumps(new_data,ensure_ascii=False))

with open("/data/hypertext/sharpwang/dataset/TTS/Genshin_Dataset/paimeng/meta_1_data.json","r") as f:
    data = json.load(f)
print(data[0])
