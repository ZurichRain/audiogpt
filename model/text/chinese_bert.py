import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("/data/hypertext/sharpwang/TTS/Bert-VITS2/bert/chinese-roberta-wwm-ext-large")
model = AutoModelForMaskedLM.from_pretrained("/data/hypertext/sharpwang/TTS/Bert-VITS2/bert/chinese-roberta-wwm-ext-large").to(device)

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt').to(device)
        # for i in inputs:
        #     inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res['hidden_states'][-3:-2], -1)[0].cpu()
    # print(len(word2ph))
    # print(len(text)+2)
    # print(text)
    
    assert len(word2ph) == len(text)+2
    word2phone = word2ph
    phone_level_feature = []
    if res.size()[0] != len(word2phone):
        return None
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(int(word2phone[i]), 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)


    return phone_level_feature.T

def save_audio_path(feature,audiopath):
    bert_file = os.path.join('/data/hypertext/sharpwang/dataset/TTS/yujie_wav_1/yujieid/', audiopath+'.bert.pt')
    # dirname = os.path.join(dirname, spk.split('_')[1])

    # bert_file = os.path.join(dirname, audiopath+'.bert.pt')

    torch.save(feature,bert_file)

if __name__ == '__main__':
    filename = '/data/hypertext/sharpwang/TTS/Bert-VITS2/filelists/yujie.cleaned'

    with open(filename,'r') as f:
        filepaths_and_text = [line.strip().split('|') for line in f]
    
    skip=0
    for d in tqdm(filepaths_and_text):
        audiopath, language, text, phones, tone, word2ph = d
        word2ph = word2ph.split(' ')
        for i in range(len(word2ph)):
            word2ph[i] = int(word2ph[i]) * 2
        word2ph[0] += 1
        # print(d)
        try:
            assert len(word2ph) == len(text)+2
        except:
            skip+=1
            continue
        feature = get_bert_feature(text, word2ph)

        if feature is None:
            skip+=1
            continue
        # print(feature.tolist()[0])
        save_audio_path(feature,audiopath)
        # exit()
        # exit()
    # import torch
    print(skip)
    print(len(filepaths_and_text))

    # word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
    # word2phone = [1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1]

    # # 计算总帧数
    # total_frames = sum(word2phone)
    # print(word_level_feature.shape)
    # print(word2phone)
    # phone_level_feature = []
    # for i in range(len(word2phone)):
    #     print(word_level_feature[i].shape)

    #     # 对每个词重复word2phone[i]次
    #     repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
    #     phone_level_feature.append(repeat_feature)

    # phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # print(phone_level_feature.shape)  # torch.Size([36, 1024])

