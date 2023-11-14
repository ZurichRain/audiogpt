import os
import json
import torch
import numpy as np
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn


data_root = "/data/hypertext/sharpwang/dataset/TTS/LibriSpeech_1/train-clean-360"

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}
def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec



def get_audio(config, audiopath):
    # filename = audiopath.split("/")[-1]
    audio, sampling_rate = load_wav_to_torch(audiopath)

    if sampling_rate != config.sampling_rate:
        raise ValueError("{} {} SR doesn't match target SR".format(
            sampling_rate, config.sampling_rate))

    audio_norm = audio / config.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)

    if config.use_mel_spec_posterior:
        spec = mel_spectrogram_torch(audio_norm, config.filter_length,
            config.n_mel_channels, config.sampling_rate, config.hop_length,
            config.win_length, config.config.mel_fmin, config.config.mel_fmax, center=False)
    else:

        spec = spectrogram_torch(audio_norm, config.filter_length,
            config.sampling_rate, config.hop_length, config.win_length,
            center=False)
        
    spec = torch.squeeze(spec, 0)
    return spec, audio_norm


class WavConfig:
    def __init__(self,kv) -> None:
        for k,v in kv.items():
            self.__setattr__(k,v)



all_meta_data = []
skip_sample = 0

dic_config = {
    "data_root": "/data/hypertext/sharpwang/dataset/TTS/LibriSpeech_1/train-clean-360",
    "max_wav_value": 32768.0,
    "sampling_rate": 16000,
    "filter_length": 2048,
    "hop_length": 512,
    "win_length": 2048,
    "n_mel_channels": 128,
    "mel_fmin": 0.0,
    "mel_fmax": None,
    "add_blank": True,
    "n_speakers": 256,
    "cleaned_text": True,
    "spk2id": {
        "P1": 0
    },
    "audio_patch_token_id": 32000,
    "audio_start_token_id": 32001,
    "audio_end_token": 32002,
    "conversation_datasets":True,
    "use_audio_start_end": False,
    "use_decoder_audio_start_end":True,
    "max_mel_len": 512
}
config = WavConfig(dic_config)

for speaker_id in os.listdir(data_root):
    for chapter_id in os.listdir(os.path.join(data_root,speaker_id)):
        with open(os.path.join(data_root,speaker_id,chapter_id,"meta_data.json"),'r') as f:
            data=json.load(f)
        # print(data)
        for d in data:
            d["wav_path"] = os.path.join(data_root, speaker_id, chapter_id, d["id"]+".wav")
            try:
                get_audio(config, d["wav_path"])
            except Exception as e:
                skip_sample += 1
                if skip_sample == 1:
                    print(e)
                continue
            all_meta_data.append(d)
            # print(d)
            # exit()
        # exit()
print(skip_sample)
with open("/data/hypertext/sharpwang/dataset/TTS/LibriSpeech_1/train-clean-360/all_meta_data.json",'w') as f:
    f.write(json.dumps(all_meta_data, ensure_ascii=False))
