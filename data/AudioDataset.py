import os
import sys

sys.path.append("/data/hypertext/sharpwang/TTS/Audiogpt")
sys.path.append("/data/hypertext/sharpwang/TTS/")

import json
import copy

import torch
from torch.utils.data import Dataset
import transformers

from utils import *

class AudioDataset(Dataset):
    '''
        data:
            meta_data.json
            wav_lis
    '''
    def __init__(self, config, tokenizer) -> None:
        super().__init__()
        self.max_wav_value = config.max_wav_value
        self.sampling_rate = config.sampling_rate
        self.filter_length = config.filter_length
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.sampling_rate = config.sampling_rate
        self.spk_map = config.spk2id
        self.data_root = config.data_root
        self.config = config

        self.use_mel_spec_posterior = getattr(config, "use_mel_posterior_encoder", False)
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(config, "n_mel_channels", 80)
        
        self.min_text_len = getattr(config, "min_text_len", 1)
        self.max_text_len = getattr(config, "max_text_len", 300)

        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        #                     model_args.model_name_or_path,
        #                     cache_dir=training_args.cache_dir,
        #                     model_max_length=training_args.model_max_length,
        #                     padding_side="right",
        #                     use_fast=False,
        #                 )
        self.tokenizer = tokenizer

        train_data_path = config.data_root + "/meta_1_data.json"
        with open(train_data_path,'r') as f:
            self.train_data=json.load(f)
        
        # print(self.train_data[0])
        
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        train_data_new = []
        skipped = 0

        for c_data in self.train_data:

            _id, qalis, wavid = c_data["id"], c_data["qalis"], c_data["wavid"]

            audiopath = self.data_root+"/wav_lis/"+wavid+".wav"
            try:
                self.get_audio(audiopath)
            except:
                skipped += 1
                continue
            train_data_new.append(c_data)
            
        print("skipped: ", skipped, ", total: ", len(self.train_data), 'new: ', len(train_data_new))
        self.train_data = train_data_new
    

    

    def get_audio(self, audiopath):
        filename = audiopath.split("/")[-1]
        audio, sampling_rate = load_wav_to_torch(audiopath)

        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target SR".format(
                sampling_rate, self.sampling_rate))

        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")

        if not os.path.exists(self.data_root+"/spec_lis/"):
            os.makedirs(self.data_root+"/spec_lis/")
        
        spec_filepath = self.data_root+"/spec_lis/"+spec_filename

        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
            if not os.path.exists(self.data_root+"/mel_lis/"):
                os.makedirs(self.data_root+"/mel_lis/")
            spec_filepath = self.data_root+"/mel_lis/"+spec_filename

        if os.path.exists(spec_filepath):
            spec = torch.load(spec_filepath)
            # print(spec.size())
        else:
            if self.use_mel_spec_posterior:
                # if os.path.exists(filename.replace(".wav", ".spec.pt")):
                #     # spec, n_fft, num_mels, sampling_rate, fmin, fmax
                #     spec = spec_to_mel_torch(
                #         torch.load(filename.replace(".wav", ".spec.pt")), 
                #         self.filter_length, self.n_mel_channels, self.sampling_rate,
                #         self.hparams.mel_fmin, self.hparams.mel_fmax)
                spec = mel_spectrogram_torch(audio_norm, self.filter_length,
                    self.n_mel_channels, self.sampling_rate, self.hop_length,
                    self.win_length, self.config.mel_fmin, self.config.mel_fmax, center=False)
            else:
                # print(audio_norm.size())
                # try:
                #     spec = spectrogram_torch(audio_norm, self.filter_length,
                #         self.sampling_rate, self.hop_length, self.win_length,
                #         center=False)
                # except:
                #     print(spec)
                #     print(spec.size())
                spec = spectrogram_torch(audio_norm, self.filter_length,
                    self.sampling_rate, self.hop_length, self.win_length,
                    center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filepath)

        return spec, audio_norm
    
    
    def __getitem__(self, index):
        '''
            {
                id:xx
                qalis:[[q,a],[q,a]]
                wavid:xxx
            }

            {
                id:xx
                qalis:[[q,a],[q,a]]
                wavid:xxx
            }
        '''
        c_data = self.train_data[index]

        conversations = c_data["qalis"]
        
        target_ids = []
        input_ids = []

        if "wavid" in c_data:
            audiopath = self.data_root+"/wav_lis/"+c_data["wavid"]+".wav"
            spec, audio_norm = self.get_audio(audiopath)

        # for _ in range(spec.size()[1]):
        #     input_ids.append(self.tokenizer.convert_tokens_to_ids(["<audio>"])[0])
        #     target_ids.append(-100)
        

        for idx, question_answer in enumerate(conversations):
            cur_question = question_answer[0].strip()
            cur_answer = question_answer[1].strip()
            
            cur_instruct = "### Human: " + cur_question + "### Assistant: "
            all_cur_seq = cur_instruct + cur_answer + "</s>"

            # answer 内部添加 特殊token <SPST> <audio> <SPED> 
            
            

            # print(cur_instruct)
            # print(all_cur_seq)
            # print(cur_answer + "</s>")
            
            
            cur_round_instruct_ids = self.tokenizer(cur_instruct).input_ids
            cur_round_answer_ids = self.tokenizer(cur_answer+"</s>").input_ids
            # cur_round_input_ids = self.tokenizer(all_cur_seq).input_ids
            cur_round_input_ids = cur_round_instruct_ids + cur_round_answer_ids[1:]

            # print(self.tokenizer.convert_ids_to_tokens(cur_round_input_ids))
            cur_round_target_ids = copy.deepcopy(cur_round_input_ids)
            # print(self.tokenizer.convert_ids_to_tokens(ids_input))
            # print(self.tokenizer(all_cur_seq))
            # print(cura)
            # c_target = ids_input_1.clone()
            # cur_round_target_ids[:-len(cur_round_answer_ids)+1] = -100
            # print(cur_round_input_ids)
            # print(cur_round_instruct_ids)
            # print(cur_round_answer_ids)
            # import ipdb; ipdb.set_trace()
            # exit()
            
            # for c_idx in range(len(cur_round_target_ids)-(len(cur_round_answer_ids)-1)):
            #     cur_round_target_ids[c_idx] = -100
            
            for c_idx in range(len(cur_round_instruct_ids)):
                cur_round_target_ids[c_idx] = -100

            # print(cur_round_instruct_ids)
            # print(cur_round_answer_ids)
            # print(cur_round_input_ids)
            # print(cur_round_target_ids)
            
            if idx==0 :
                input_ids += cur_round_input_ids
                target_ids += cur_round_target_ids
            else:
                input_ids += cur_round_input_ids[1:]
                target_ids += cur_round_target_ids[1:]
        
        # print(input_ids)
        # print(target_ids)
        # exit()
        
        audio_token_id = self.tokenizer.convert_tokens_to_ids(["<audio>"])[0]

        aft_input_ids = []
        aft_target_ids = []
        audio_st = []
        audio_idx_cnt = 0
        for idx, c_id in enumerate(input_ids):
            if c_id == audio_token_id:
                aft_input_ids += [ audio_token_id ] * (spec.size()[1])
                # target_ids = target_ids[:idx]+[-100]*(spec.size()[1])+target_ids[idx+1:]
                aft_target_ids += [-100]*(spec.size()[1])
                audio_st.append(audio_idx_cnt)
                audio_idx_cnt+=spec.size()[1]
            else:
                aft_input_ids.append(c_id)
                aft_target_ids.append(target_ids[idx])
                audio_idx_cnt+=1
        assert len(aft_target_ids) == len(aft_input_ids), print(len(aft_target_ids)," ",len(aft_input_ids))
        
            # print(spec.size())
            # print(audio_norm.size())
        # print(spec.size())
        
        # return None
        # test_input = torch.tensor(input_ids)
        # print(test_input.size())
        # print(torch.where(test_input == torch.tensor(32003)))
        # audio_patch_tokens = torch.where(test_input == torch.tensor(32003))[0]
        # print(test_input[:audio_patch_tokens])
        # print(test_input[audio_patch_tokens+1:])
        # print(audio_patch_tokens)
        # print(torch.tensor(target_ids))
        # exit()
        # import ipdb;ipdb.set_trace()
        
        # result = {
        #     "input_ids": torch.tensor(aft_input_ids),
        #     "labels": torch.tensor(target_ids),
        #     "audios": [spec.transpose(0, 1)], # len(<audio_path>),spec_len,1025
        #     "audio_lens": [spec.size()[1]], # len(<audio_path>)
        #     "audio_st": [audio_st],
        # }
        result = dict(
            input_ids=torch.tensor(aft_input_ids),
            labels=torch.tensor(aft_target_ids),
            audios=[spec.transpose(0, 1)], # len(<audio_path>),spec_len,1025
            audio_lens=[spec.size()[1]], # len(<audio_path>)
            audio_start= audio_st,
        )
        # print(result)
        return result

    def __len__(self):
        return len(self.train_data)
    
if __name__ == "__main__":
    from config.configs import dataConfig
    
    with open("/data/hypertext/sharpwang/TTS/Audiogpt/config/config.json","r") as f:
        config_dict = json.load(f) 
    
    config = dataConfig(config_dict["data"])
    

    model_name_or_path = "/data/hypertext/sharpwang/lessIsMore/mmgpt/checkpoints/mmgpt-7b-stage1-cc_sbu_558k_v0"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
                            model_name_or_path,
                            cache_dir= None,
                            model_max_length= 512,
                            padding_side= "right",
                            use_fast= False,
                        )
    # tokenizer.add_special_tokens({
    #             "eos_token": "</s>",
    #             "bos_token": "</s>",
    #             "unk_token": "<unk>",
    #         })
    tokenizer.pad_token = tokenizer.unk_token
    # tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
    
    
    print(len(tokenizer))
    
    tokenizer.add_tokens("<audio>",special_tokens=True)
    tokenizer.add_tokens("<SPST>",special_tokens=True) # 生成 audio的时候使用
    tokenizer.add_tokens("<SPED>",special_tokens=True)


    print(len(tokenizer))
    dataset = AudioDataset(config,tokenizer)
    from torch.utils.data import DataLoader
    from Audiogpt.data.AudioDatacollector import DataCollatorForSupervisedDataset
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=data_collator)
    # idx = 0
    # for data in dataset:
    #     print(data)
    #     if idx>=0:
    #         exit()
    #     idx+=1
    for d in dataloader:
        print(d)
        exit()
    