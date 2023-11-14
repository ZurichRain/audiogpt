import os
import json


data_root = "/data/hypertext/sharpwang/dataset/TTS/LibriSpeech_1/train-clean-360"

for speaker_id in os.listdir(data_root):
    for chapter_id in os.listdir(os.path.join(data_root,speaker_id)):
        with open(os.path.join(data_root,speaker_id,chapter_id,"meta_data.json"),'r') as f:
            data=json.load(f)
        for d in data:
            print(d)
            exit()

