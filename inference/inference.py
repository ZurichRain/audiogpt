# /data/hypertext/sharpwang/TTS/Audiogpt/model/AudiogptBase.py
import sys
sys.path.append("/data/hypertext/sharpwang/TTS/")

import torch
from Audiogpt.model.AudiogptBase import AudioGPTLlamaForCausalLM
from transformers import AutoTokenizer
from Audiogpt.data.utils import *
from config.configs import dataConfig


DEFAULT_AUDIO_PATCH_TOKEN = "<audio>"


def load_model(model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AudioGPTLlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda")

    tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)

    return model, tokenizer

def get_audio(config, audiopath,
              use_mel_spec_posterior = False,
              ):
    root_path = "/".join(audiopath.split("/")[:-1])
    filename = audiopath.split("/")[-1]
    audio, sampling_rate = load_wav_to_torch(audiopath)

    if sampling_rate != config.sampling_rate:
        raise ValueError("{} {} SR doesn't match target SR".format(
            sampling_rate, config.sampling_rate))

    audio_norm = audio / config.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    spec_filename = filename.replace(".wav", ".spec.pt")

    if not os.path.exists(root_path+"/spec_lis/"):
        os.makedirs(root_path+"/spec_lis/")
    
    spec_filepath = root_path+"/spec_lis/"+spec_filename

    if use_mel_spec_posterior:
        spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        if not os.path.exists(root_path+"/mel_lis/"):
            os.makedirs(root_path+"/mel_lis/")
        spec_filepath = root_path+"/mel_lis/"+spec_filename

    if os.path.exists(spec_filepath):
        spec = torch.load(spec_filepath)
        # print(spec.size())
    else:
        if use_mel_spec_posterior:
            # if os.path.exists(filename.replace(".wav", ".spec.pt")):
            #     # spec, n_fft, num_mels, sampling_rate, fmin, fmax
            #     spec = spec_to_mel_torch(
            #         torch.load(filename.replace(".wav", ".spec.pt")), 
            #         self.filter_length, self.n_mel_channels, self.sampling_rate,
            #         self.hparams.mel_fmin, self.hparams.mel_fmax)
            spec = mel_spectrogram_torch(audio_norm, config.filter_length,
                config.n_mel_channels, config.sampling_rate, config.hop_length,
                config.win_length, config.config.mel_fmin, config.config.mel_fmax, center=False)
        else:
            # print(audio_norm.size())
            # try:
            #     spec = spectrogram_torch(audio_norm, self.filter_length,
            #         self.sampling_rate, self.hop_length, self.win_length,
            #         center=False)
            # except:
            #     print(spec)
            #     print(spec.size())
            spec = spectrogram_torch(audio_norm, config.filter_length,
                config.sampling_rate, config.hop_length, config.win_length,
                center=False)
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_filepath)

    return spec, audio_norm


def get_input_for_model(config_data, wav_path, prompt, tokenizer):

    input_ids = []
    # print(config_data)
    spec, audio_norm = get_audio(config_data, wav_path)
    instruct = "Human: " + prompt + "Assistant: "
    instruct_ids = tokenizer(instruct).input_ids
    # all_instr  = "Human: " + prompt + "Assistant: " + "不得不说，这段文字所讲述的是：咦，蒂玛乌斯先生今天好像不在…</s>" 
    # print(tokenizer("不得不说，这段文字所讲述的是：咦，蒂玛乌斯先生今天好像不在…</s>").input_ids)
    # all_instr = tokenizer(all_instr).input_ids
    
    # print(all_instr)

    # print(instruct_ids)

    audio_token_id = tokenizer.convert_tokens_to_ids(["<audio>"])[0]
    aft_input_ids = []
    audio_st = []
    audio_idx_cnt = 0
    for idx, c_id in enumerate(instruct_ids):
        if c_id == audio_token_id:
            aft_input_ids += [ audio_token_id ] * (spec.size()[1])
            audio_st.append(audio_idx_cnt)
            audio_idx_cnt+=spec.size()[1]
        else:
            aft_input_ids.append(c_id)
            audio_idx_cnt+=1

    input_ids= torch.tensor([aft_input_ids]).to("cuda")
    # print(input_ids)
    # print(spec.dtype)
    audios=[[spec.transpose(0, 1).to("cuda", dtype=torch.bfloat16)]]
    audio_lens=[[spec.size()[1]]]
    audio_start= [audio_st]
    return input_ids, audios, audio_lens, audio_start


wav_path = "/data/hypertext/sharpwang/TTS/Audiogpt/inference/vo_ABDLQ002_1_paimon_04.wav"
prompt = "这段音频说了啥？<audio>。"
model_path = "/data/hypertext/sharpwang/TTS/Audiogpt/checkpoints/Audiogpt-7b-stage1-paimeng_v1_epoch_10_lr_2e-5_seed_1226_sep_by_->"


with open("/data/hypertext/sharpwang/TTS/Audiogpt/config/config.json","r") as f:
        config_dict = json.load(f) 

data_config = dataConfig(config_dict["data"])
model_config = dataConfig(config_dict["model"])
train_config = dataConfig(config_dict["train"])


model, tokenizer = load_model(model_path)
model.init_audio_config(data_config)
# Dataset

input_ids, audios, audio_lens, audio_start = get_input_for_model(data_config, wav_path, prompt, tokenizer)

ans = model.inference(
    input_ids=input_ids,
    audios=audios,
    audio_lens= audio_lens,
    audio_start=audio_start,
    tokenizer = tokenizer,
    temperature = 0
)

print(ans)
with open("ans.txt",'w') as f:
    f.write(ans)




