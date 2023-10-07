# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
sys.path.append("/data/hypertext/sharpwang/TTS/Audiogpt")
sys.path.append("/data/hypertext/sharpwang/TTS/")
import logging
import pathlib
import json
import torch
import transformers

# from mmgpt.train.trainer import MMGPTTrainer
# from mmgpt.train.trainer_vit_llrd import MMGPTTrainer
from Audiogpt.train.trainer import AudioGPTTrainer
from Audiogpt.model.AudiogptBase import AudioGPTLlamaForCausalLM
# from Audiogpt.data import make_supervised_data_module
# from Audiogpt.utils.arguments import *
# from Audiogpt.utils.constants import *
# from Audiogpt.utils.utils import smart_tokenizer_and_embedding_resize
from Audiogpt.data.AudioDatacollector import DataCollatorForSupervisedDataset
from config.configs import dataConfig

from Audiogpt.model.AudiogptBase import *
from Audiogpt.data.AudioDataset import AudioDataset



def make_supervised_data_module(model_args, tokenizer, data_args):
    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    # multimodal_cfg = dict(
    #     use_audio_start_end=data_args.use_audio_start_end,
    # )
    # from mmgpt.data.interleave_dataset import InterleaveDataset
    

    # interleave_dataset = InterleaveDataset(
    #     tokenizer=tokenizer,
    #     datasets=data_args.interleave_datasets,
    #     multimodal_cfg=multimodal_cfg
    # ) if data_args.interleave_datasets else None
    conversation_dataset = AudioDataset(
        config=data_args,
        tokenizer=tokenizer,
    ) if data_args.conversation_datasets else None

    # if interleave_dataset is not None and conversation_dataset is not None:
    #     train_dataset = torch.utils.data.ConcatDataset([interleave_dataset, conversation_dataset])
    # elif interleave_dataset is not None:
    #     train_dataset = interleave_dataset
    # elif conversation_dataset is not None:
    #     train_dataset = conversation_dataset
    # else:
    #     raise ValueError("Training Dataset is undefined!")
    train_dataset = conversation_dataset
    
    print(f'After processing, totally {len(train_dataset)} samples are involved.')

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def train():

    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    
    with open("/data/hypertext/sharpwang/TTS/Audiogpt/config/config.json","r") as f:
        config_dict = json.load(f) 

    parser = transformers.HfArgumentParser((transformers.TrainingArguments))
    TrainArg_for_Trainer = parser.parse_args_into_dataclasses()[0]
    # print(TrainArg_for_Trainer)
    
    data_config = dataConfig(config_dict["data"])
    model_config = dataConfig(config_dict["model"])
    train_config = dataConfig(config_dict["train"])

    model = AudioGPTLlamaForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=model_config.cache_dir,
        ignore_mismatched_sizes=True
    )
    model.init_audio_config(data_config)
    

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=train_config.cache_dir,
        model_max_length=model_config.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # pad 没有定义
    tokenizer.pad_token = tokenizer.unk_token

    # if data_args.conversation_version == "v0" or "models--decapoda-research--llama-7b-hf" in model_args.model_name_or_path:
    #     if tokenizer.pad_token is None:
    #         smart_tokenizer_and_embedding_resize(
    #             special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
    #             tokenizer=tokenizer,
    #             model=model,
    #         )
    #     if "llama" in model_args.model_name_or_path:
    #         tokenizer.add_special_tokens({
    #             "eos_token": DEFAULT_EOS_TOKEN,
    #             "bos_token": DEFAULT_BOS_TOKEN,
    #             "unk_token": DEFAULT_UNK_TOKEN,
    #         })
    #     print('loading v0 tokenizer.')
    # else:
    #     tokenizer.pad_token = tokenizer.unk_token
    #     print('loading v1 tokenizer.')
    #     print(tokenizer)

    dtype = torch.float32
    if train_config.fp16:
        dtype = torch.float16
    if train_config.bf16:
        dtype = torch.bfloat16

    train_config.__setattr__("device", "cuda")

    audio_encoder_dict = model.get_model().initialize_audio_modules(
        AudioEncoderPath=model_config.AudioEncoderPath,
        pretrained_stage1_model=model_config.pretrained_stage1_model,
        freeze_audio_encoder=model_config.freeze_audio_encoder,
        use_audio_start_end=model_config.use_audio_start_end,
        audio_select_layer=model_config.audio_select_layer,
        dtype=dtype,
        device=train_config.device
    )

    model.initialize_audio_tokenizer(
        tokenizer=tokenizer, 
        freeze_lm_model=model_config.freeze_lm_model, 
        pretrained_stage1_model=model_config.pretrained_stage1_model,
        device=train_config.device,
    )

    model.to(dtype=dtype, device=train_config.device)

    # data_args.image_token_len = vision_tower_dict['image_token_len']
    # data_args.image_processor = vision_tower_dict['image_processor']
    # data_config.use_im_start_end = audio_encoder_dict.use_im_start_end

    # mixed relation, to be fixed
    if model_config.freeze_lm_model:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        for p in model.get_input_embeddings().parameters():
            p.requires_grad = True

        # if not model_config.freeze_audio_encoder:
        #     model.get_model().AudioEncoder.requires_grad_(True)
        #     # for i in range(20):
        #     #     model.get_model().vision_tower.vision_model.encoder.layers[i].requires_grad_(False)
        #     model.get_model().AudioEncoder.vision_model.encoder.layers[-1].requires_grad_(False)
        #     # model.get_model().vision_tower.vision_model.embeddings.requires_grad_(False)
        #     # model.get_model().vision_tower.vision_model.pre_layrnorm.requires_grad_(False)
        #     model.get_model().AudioEncoder.vision_model.post_layernorm.requires_grad_(False)

            # for n, p in model.named_parameters():
            #     print(n, p.requires_grad)
                
    params_grad = [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of Mapping Trainable Parameters: {sum(params_grad) / (1 << 20):.2f} M")

    # params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    # if len(params_no_grad) > 0:
    #     if training_args.fsdp is not None and len(training_args.fsdp) > 0:
    #         if len(params_no_grad) < 10:
    #             print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
    #         else:
    #             print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
    #         print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
    #         print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

    #         from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    #         def patch_FSDP_use_orig_params(func):
    #             def wrap_func(*args, **kwargs):
    #                 use_orig_params = kwargs.pop('use_orig_params', True)
    #                 return func(*args, **kwargs, use_orig_params=use_orig_params)
    #             return wrap_func

    #         FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)


    data_module = make_supervised_data_module(
        model_config,
        tokenizer=tokenizer, 
        data_args=data_config
    )

    trainer = AudioGPTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainArg_for_Trainer,
        **data_module)

    if list(pathlib.Path(TrainArg_for_Trainer.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    trainer._safe_save(output_dir=TrainArg_for_Trainer.output_dir)


if __name__ == "__main__":
    train()
