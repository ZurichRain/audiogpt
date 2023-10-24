#    Copyright 2023 Haotian Liu
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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

# from Audiogpt.utils.constants import *
from Audiogpt.model.AudioEncoder import PosteriorEncoder

import torchaudio
import transformers

from utils import *

from audio_decoder import SynthesizerTrn


DEFAULT_AUDIO_PATCH_TOKEN = "<audio>"
DEFAULT_AUDIO_START_TOKEN = "<SPST>"
DEFAULT_AUDIO_END_TOKEN = "<SPED>"
DEFAULT_AUDIO_DECODER_START_TOKEN = "<DECODER_ST>"
DEFAULT_AUDIO_DECODER_END_TOKEN = "<DECODER_ED>"



class AudioLlamaConfig(LlamaConfig):
    model_type = "audio_llama"


class AudioLlamaModel(LlamaModel):
    config_class = AudioLlamaConfig

    def __init__(self, config: LlamaConfig,
                    AudioEncoderConfig = None,
                    AudioDecoderConfig = None):
        super(AudioLlamaModel, self).__init__(config)

        self.AudioEncoderConfig = AudioEncoderConfig
        self.AudioDecoderConfig = AudioDecoderConfig
        
        if self.AudioDecoderConfig is not None:
            self.AudioEncoder = PosteriorEncoder.load_state_dict("bert vits2 path")

        # out_channels
        if self.AudioDecoderConfig is not None:
            self.mm_projector = nn.Linear(self.AudioDecoderConfig.out_channels, self.config.hidden_size)
        else:
            # 1025 是因为spec的隐藏层维度是 1025 
            self.mm_projector = nn.Linear(1025, self.config.hidden_size)
    def init_audio_config(self,audio_config):
        self.audio_config = audio_config
    
    def initialize_audio_modules(
        self, 
        AudioEncoderPath = None,
        pretrained_stage1_model = None,
        freeze_audio_encoder = False,
        use_audio_start_end = False,
        audio_select_layer = -1,
        dtype=torch.float16,
        device="cuda"
    ):
        
        if not hasattr(self, 'audio_encoder') and AudioEncoderPath is not None and self.AudioEncoderConfig:
            self.AudioEncoder = PosteriorEncoder.load_state_dict(AudioEncoderPath)

            self.AudioEncoder = self.AudioEncoder.to(dtype=dtype, device=device)

        # vision_config = self.vision_tower.config

        if not hasattr(self, 'mm_projector') and self.AudioEncoderConfig:
            self.mm_projector = nn.Linear(self.AudioDecoderConfig.out_channels, self.config.hidden_size)
        elif not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(1025, self.config.hidden_size)

        
        self.mm_projector = self.mm_projector.to(dtype=dtype, device=device)


        if pretrained_stage1_model is not None:
            stage1_weights = torch.load(pretrained_stage1_model, map_location='cpu')

            mm_projector_weight = {
                'weight': stage1_weights['model.mm_projector.weight'],
                'bias': stage1_weights['model.mm_projector.bias'],
            }
            self.mm_projector.load_state_dict(mm_projector_weight)

            if 'model.vision_tower.vision_model.embeddings.class_embedding' in stage1_weights:
                print("vision tower weights are loaded!")
                self.vision_tower.load_state_dict({k[19:]: v for k, v in stage1_weights.items() if 'vision_tower' in k}, strict=True)

        # image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        # self.config.vision_tower = vision_tower
        # self.config.image_token_len = image_token_len
        self.audio_config.use_audio_start_end = use_audio_start_end
        self.audio_config.audio_select_layer = audio_select_layer
        self.audio_config.freeze_audio_encoder = freeze_audio_encoder
        
        return self.config

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        
        audios: Optional[torch.FloatTensor] = None,
        audio_lens: Optional[torch.FloatTensor] = None,
        audio_start = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        '''
            input_ids: 一个batch的text
            audios: 是一个batch的wav 
        '''

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        if orig_embeds_params is not None:
            with torch.no_grad():
                self.get_input_embeddings().weight[:-self.num_new_tokens] = orig_embeds_params[:-self.num_new_tokens].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # print(inputs_embeds)

        # TODO 这里可有可无
        audio_encoder = getattr(self, 'audio_encoder', None)

        # get audio embedding and text embedding
        use_audio_start_end = getattr(self.audio_config, "use_audio_start_end", False)
        audio_patch_token = getattr(self.audio_config, "audio_patch_token_id", -1)
        audio_start_token = getattr(self.audio_config, "audio_start_token_id", -1)
        audio_end_token = getattr(self.audio_config, "audio_end_token_id", -1)
        freeze_audio_encoder = getattr(self.audio_config, "freeze_audio_encoder", False)

        
        if audios is not None:
            audio_features=audios
            if audio_encoder is not None :
                with torch.set_grad_enabled(not freeze_audio_encoder):
                    audio_encoder_out = audio_encoder(audios, audio_lens, output_hidden_states=True)
                    audio_features = audio_encoder_out[0]
            
            audio_features_text_emb = []
            for example_audios in audio_features:
                exmaple_audios_text_emb = []
                for each_audio_in_example in example_audios:
                    exmaple_audios_text_emb.append(self.mm_projector(each_audio_in_example))
                audio_features_text_emb.append(exmaple_audios_text_emb)

            max_len = -1 
            new_input_embeds = []

            for cur_text_ids, cur_text_embeds, cur_audio_features, cur_audio_sts in zip(input_ids, inputs_embeds, audio_features_text_emb, audio_start):

                # import ipdb;ipdb.set_trace()
                if (cur_text_ids == audio_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    new_input_embeds.append(cur_text_embeds)
                    continue

                tidx=0
                for  per_cur_audio_features, audio_st in zip(cur_audio_features, cur_audio_sts):
                    if tidx != len(cur_audio_sts)-1:
                        cur_text_embeds = torch.cat(
                                (
                                    cur_text_embeds[:audio_st], 
                                    per_cur_audio_features, 
                                    cur_text_embeds[audio_st + per_cur_audio_features.size()[0]: cur_audio_sts[tidx+1]]
                                ), 
                                dim=0
                            )
                        assert (cur_text_ids[:audio_st] == audio_patch_token).sum() == 0 and \
                             (cur_text_ids[audio_st + per_cur_audio_features.size()[0]: cur_audio_sts[tidx+1]] == audio_patch_token).sum() == 0
                    else:
                        cur_text_embeds = torch.cat(
                                (
                                    cur_text_embeds[:audio_st], 
                                    per_cur_audio_features, 
                                    cur_text_embeds[audio_st + per_cur_audio_features.size()[0]:]
                                ), 
                                dim=0
                            )
                        # print(cur_text_ids[audio_st:audio_st + per_cur_audio_features.size()[0]])
                        assert torch.all(cur_text_ids[audio_st:audio_st + per_cur_audio_features.size()[0]].eq(audio_patch_token))
                        assert (cur_text_ids[:audio_st] == audio_patch_token).sum() == 0 and \
                             (cur_text_ids[audio_st + per_cur_audio_features.size()[0]:] == audio_patch_token).sum() == 0
                              
                        # print(cur_text_ids[:audio_st])
                        # print(cur_text_ids[audio_st + per_cur_audio_features.size()[0]:])
                        # import ipdb; ipdb.set_trace()
                        
                    max_len = max(max_len, cur_text_embeds.size()[0])
                    tidx += 1
                        # assert cur_text_ids.size()[0]+per_cur_audio_features.size()[0]-1== cur_text_embeds.size()[0],\
                        #     print(cur_text_embeds.size()[0]," ",cur_text_ids.size()[0]+per_cur_audio_features.size()[0]-1)
                    # print(cur_text_ids.size()[0]+per_cur_audio_features.size()[0]-1)
                    # print(cur_text_embeds.size())
                    
                    

                # cur_all_input_embedding = [cur_text_embeds] 

                if use_audio_start_end:
                    # TODO
                    raise NotImplementedError
                    # pass
                
                # cur_all_input_embedding.append(cur_audio_features)
                new_input_embeds.append(cur_text_embeds)
            # print(max_len)
            # for bidx in range(len(new_input_embeds)):
            #     if new_input_embeds[bidx].size()[0] < max_len:
            #         new_input_embeds[bidx]=torch.cat(
            #             (
            #                 new_input_embeds[bidx],
            #                 torch.zeros(max_len-new_input_embeds[bidx].size()[0], new_input_embeds[bidx].size()[1]).to("cuda")
            #             ),
            #             dim=0
            #         )
                
            inputs_embeds=torch.stack(new_input_embeds, dim=0)

        
        # print(inputs_embeds.size())

        # wav_feature
        # if audio_encoder is not None and (input_ids.shape[1] != 1 or self.training) and audios is not None:
            
        #     assert audio_lens is not None
            
        #     use_audio_start_end = getattr(self.config, "use_audio_start_end", -1)
        #     audio_patch_token = getattr(self.config, "audio_patch_token_id", -1)
        #     audio_start_token = getattr(self.config, "audio_start_token_id", -1)
        #     audio_end_token = getattr(self.config, "audio_end_token_id", -1)
        #     freeze_audio_encoder = getattr(self.config, "freeze_audio_encoder", False)

        #     with torch.set_grad_enabled(not freeze_audio_encoder):
        #         audio_encoder_out = audio_encoder(audios, audio_lens, output_hidden_states=True)
        #         audio_features = audio_encoder_out[0]
            
        #     # dummy_audio_features = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        #     # dummy_image_features = self.mm_projector(dummy_image_features)
        #     audio_features = self.mm_projector(audio_features)

        #     new_input_embeds = []
        #     for cur_text_ids, cur_text_embeds, cur_audio_features in zip(input_ids, inputs_embeds, audio_features):
        #         if (cur_text_ids == audio_patch_token).sum() == 0:
        #             # multimodal LLM, but the current sample is not multimodal
        #             new_input_embeds.append(cur_text_embeds)
        #             continue

        #         audio_patch_tokens = torch.where(cur_text_ids == audio_patch_token)[0]
                
        #         for audio_patch_token_pos, per_cur_audio_features in zip(audio_patch_tokens, cur_audio_features):
        #             # cur_text_embeds
                    
        #             cur_text_embeds = torch.cat(
        #                     (
        #                         cur_text_embeds[:audio_patch_token_pos], 
        #                         per_cur_audio_features, 
        #                         cur_text_embeds[audio_patch_token_pos + 1:]
        #                     ), 
        #                     dim=0
        #                 )

        #         # cur_all_input_embedding = [cur_text_embeds]

        #         if use_audio_start_end:
        #             # 这里可能需要 思考 如何加入新的token emb
        #             # TODO
        #             raise NotImplementedError
        #             # pass
                
        #         # cur_all_input_embedding.append(cur_audio_features)
        #         new_input_embeds.append(cur_text_embeds)

        #     inputs_embeds=torch.stack(new_input_embeds, dim=0)
        # else:
        #     raise NotImplementedError

        return super(AudioLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        
        

        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            assert type(images) is list, ValueError("To fit both interleave and conversation, images must be list of batches of images")
            
            use_im_start_end = getattr(self.config, "use_im_start_end", -1)
            vision_select_layer = getattr(self.config, "vision_select_layer", -1)
            im_patch_token = getattr(self.config, "im_patch_token", -1)
            im_start_token = getattr(self.config, "im_start_token", -1)
            im_end_token = getattr(self.config, "im_end_token", -1)
            freeze_vision_tower = getattr(self.config, "freeze_vision_tower", False)

            with torch.set_grad_enabled(not freeze_vision_tower):
                image_features = []
                for image in images:
                    image_forward_out = vision_tower(image, output_hidden_states=True)
                    select_hidden_state = image_forward_out.hidden_states[vision_select_layer]
                    image_feature = select_hidden_state[:, 1:]
                    image_features.append(image_feature)

            if type(images) is list:
                image_features = [self.mm_projector(image_feature) for image_feature in image_features]
            else:
                # image_features = self.mm_projector(image_features)
                raise NotImplementedError
            
            # dummy_image_features = torch.zeros(256, 1664, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, image_features):
                if (cur_input_ids == im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue

                if use_im_start_end:
                    if (cur_input_ids == im_start_token).sum() != (cur_input_ids == im_end_token).sum():
                        raise ValueError(f"The number of image start tokens ({(cur_input_ids == im_start_token).sum()}) and image end tokens ({(cur_input_ids == im_end_token).sum()}) should be the same.")
                    
                    image_start_tokens = torch.where(cur_input_ids == im_start_token)[0]
                    for image_start_token_pos, per_cur_image_features in zip(image_start_tokens, cur_image_features):
                        per_cur_image_features = per_cur_image_features.to(device=cur_input_embeds.device)
                        num_patches = per_cur_image_features.shape[0]

                        if cur_input_ids[image_start_token_pos + num_patches + 1] != im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        
                        # if orig_embeds_params is not None:
                        #     cur_new_input_embeds = torch.cat(
                        #         (
                        #             cur_input_embeds[:image_start_token_pos].detach(), 
                        #             cur_input_embeds[image_start_token_pos:image_start_token_pos+1], 
                        #             per_cur_image_features, 
                        #             cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], 
                        #             cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()
                        #         ), 
                        #         dim=0
                        #     )
                        # else:
                        cur_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:image_start_token_pos+1], 
                                per_cur_image_features, 
                                cur_input_embeds[image_start_token_pos + num_patches + 1:]
                            ), 
                            dim=0
                        )

                    new_input_embeds.append(cur_input_embeds)
                else:
                    raise NotImplementedError
                    # cur_image_features = image_features[cur_image_idx]
                    # num_patches = cur_image_features.shape[0]
                    # if (cur_input_ids == im_patch_token).sum() != num_patches:
                    #     raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                    # masked_indices = torch.where(cur_input_ids == im_patch_token)[0]
                    # mask_index_start = masked_indices[0]
                    # if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    #     raise ValueError("The image patch tokens should be consecutive.")
                    # if orig_embeds_params is not None:
                    #     cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_image_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    # else:
                    #     cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    # new_input_embeds.append(cur_new_input_embeds)
                    # cur_image_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(AudioLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class AudioGPTLlamaForCausalLM(LlamaForCausalLM):
    config_class = AudioLlamaConfig

    def __init__(self, config, audio_decoder_config=None, audio_decoder_model_config=None):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = AudioLlamaModel(config)
        # 这里建立一个audio decoder config
        if audio_decoder_config:
            self.audio_decoder = SynthesizerTrn(
                len(audio_decoder_config.symbols),
                audio_decoder_config.filter_length // 2 + 1,
                audio_decoder_config.segment_size // audio_decoder_config.hop_length,
                n_speakers=audio_decoder_config.n_speakers,
                **audio_decoder_model_config
            )
            load_checkpoint(audio_decoder_config.ckp_path, self.audio_decoder, None,skip_optimizer=True)

            self.audio_decoder_projector_layer=nn.Linear(config.hidden_state_size,\
                                                        audio_decoder_model_config.input_channal_size, bias=False)
        

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def init_audio_config(self,audio_config):
        self.model.init_audio_config(audio_config)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.FloatTensor] = None,
        audio_lens: Optional[torch.FloatTensor] = None,
        audio_start = None,
        y_mel= None,
        audio_decoder_st=None,
        audio_decoder_ed=None, 


        AudioSPSTED = None,
        AudioAnswerEmbedding = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            audios = audios,
            audio_lens = audio_lens,
            audio_start = audio_start
        )

        hidden_states = outputs[0]

    
        # 设置两个损失
        logits = self.lm_head(hidden_states)

        # 传递参数的时候传递一个audio st, audio ed
        # 从hidden state 出发 使用一个adapter到decoder的输入 hidden_state
        
        loss = None
        if y_mel:
            audio_hidden_states = hidden_states[:,audio_decoder_st:audio_decoder_ed,:]
            audio_hidden_states = self.audio_decoder_projector_layer(audio_hidden_states)
            y_hat_mel = self.audio_decoder(audio_hidden_states)
            loss_mel = F.l1_loss(y_mel, y_hat_mel)
            loss = loss_mel
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            if y_mel:
                loss += loss_fct(shift_logits, shift_labels)
            else:
                loss = loss_fct(shift_logits, shift_labels)

        # TODO 定位<SPGEN>的位置 然后计算Embedding的相似度
        # AudioSPSTED 记录GEN的位置 batchsize, span_size, 2
        # AudioAnswerEmbedding 记录ans中的每一段audio的表示 batchsize, span_size, audiolen, embsize
        # PredictedAudioAnswerEmbedding = hidden_states[:,] # batchsize, span_size, audiolen, embsize
        
        # if AudioAnswerEmbedding is not None:
        #     for batchidx in range(AudioAnswerEmbedding.size()[0]):
        #         for spanidx in range(AudioAnswerEmbedding.size()[1]):
        #             for audioidx in range(AudioAnswerEmbedding.size()[2]):
        #                 c_st_idx = AudioSPSTED[batchidx][spanidx][0]
        #                 loss += loss_fct(hidden_states[batchidx][spanidx][c_st_idx+audioidx],\
        #                                  AudioAnswerEmbedding[batchidx][spanidx][audioidx])
        

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_audio_tokenizer(
        self, 
        tokenizer, 
        freeze_lm_model=False, 
        pretrained_stage1_model=None,
        device="cuda"
    ):
        config = self.get_model().audio_config

        # add image patch token <image>
        tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        config.audio_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_AUDIO_PATCH_TOKEN])[0]
        

        # add image start token <im_start> and end token <im_end>
        if config.use_decoder_audio_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_AUDIO_DECODER_START_TOKEN, DEFAULT_AUDIO_DECODER_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            config.audio_start_token, config.audio_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_AUDIO_DECODER_START_TOKEN,\
                                                                                                 DEFAULT_AUDIO_DECODER_END_TOKEN])
            
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if freeze_lm_model:
                self.get_model().num_new_tokens = num_new_tokens
                self.get_model().orig_embeds_params = self.get_input_embeddings().weight.data.clone().to(device=device)
        
        if config.use_audio_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            config.audio_start_token, config.audio_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN])
            
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if freeze_lm_model:
                self.get_model().num_new_tokens = num_new_tokens
                self.get_model().orig_embeds_params = self.get_input_embeddings().weight.data.clone().to(device=device)

            if pretrained_stage1_model:
                stage1_weights = torch.load(pretrained_stage1_model, map_location='cpu')
                embed_tokens_weight = stage1_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

    def inference(
            self, 
            input_ids: torch.LongTensor = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            
            audios: Optional[torch.FloatTensor] = None,
            audio_lens: Optional[torch.FloatTensor] = None,
            audio_start = None,
            return_dict: Optional[bool] = None,

            temperature: Optional[float] = 0.9,
            max_new_tokens: Optional[int] = 256,
            stop_str: Optional[str] = "</s>",
            stop_idx: Optional[int] = 2,

            tokenizer: transformers.AutoTokenizer = None,

        ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.eval()

        # self.model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        #     audios = audios,
        #     audio_lens = audio_lens,
        #     audio_start = audio_start
        # )
        ans_str = ""
        with torch.no_grad():
            output_ids = []
            pred_ids = []
            past_key_values = None
            for i in range(max_new_tokens):
                if i == 0:
                    outputs = self.model(
                        input_ids=input_ids,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict= True,
                        audios = audios,
                        audio_lens = audio_lens,
                        audio_start = audio_start
                    )

                    hidden_states = outputs[0]
                    logits = self.lm_head(hidden_states)

                    past_key_values = outputs.past_key_values
                else:
                    # attention_mask = torch.ones(
                    #     1, past_key_values[0][0].shape[-2] + 1, device="cuda")
                    # attention_mask=attention_mask,
                    outputs = self.model(input_ids=torch.as_tensor([[token]], device="cuda"),
                                use_cache=True,
                                past_key_values=past_key_values)
                    
                    hidden_states = outputs[0]
                    logits = self.lm_head(hidden_states)

                    past_key_values = outputs.past_key_values

                last_token_logits = logits[0][-1]
                if temperature < 1e-4:
                    token = int(torch.argmax(last_token_logits))
                else:
                    probs = torch.softmax(last_token_logits / temperature, dim=-1)
                    token = int(torch.multinomial(probs, num_samples=1))

                output_ids.append(token)
                pred_ids.append(token)

                if stop_idx is not None and token == stop_idx:
                    stopped = True
                elif token == tokenizer.eos_token_id:
                    stopped = True
                else:
                    stopped = False

                if i == max_new_tokens - 1 or stopped:
                    cur_out = tokenizer.decode(pred_ids, skip_special_tokens=True)
                    pos = cur_out.rfind(stop_str)
                    if pos != -1:
                        cur_out = cur_out[:pos]
                        stopped = True
                    output = cur_out
                    ans_str += cur_out

                    # ret = {
                    #     "text": output,
                    #     "error_code": 0,
                    # }
                    # yield json.dumps(ret).encode() + b"\0"
                    print(output)

                if stopped:
                    break

            if past_key_values is not None:
                del past_key_values
        return ans_str


AutoConfig.register("audio_llama", AudioLlamaConfig)
AutoModelForCausalLM.register(AudioLlamaConfig, AudioGPTLlamaForCausalLM)
