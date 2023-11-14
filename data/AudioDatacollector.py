from dataclasses import dataclass
import transformers
import torch

@dataclass
class DataCollatorForSupervisedDataset(object):
    def __init__(self,tokenizer) -> None:
        # pass
        self.tokenizer = tokenizer

    def __call__(self, instances):
        # print(instances)

        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        if instances[0]["y_mel"] is not None:
            y_mel = [instance["y_mel"].unsqueeze(0) for instance in instances]
            audio_decoder_st = [instance["audio_decoder_st"] for instance in instances]
            audio_decoder_ed = [instance["audio_decoder_ed"] for instance in instances]
        else:
            y_mel = None
            audio_decoder_st = None
            audio_decoder_ed = None

        if instances[0]['input_audios'] is not None:
            input_audios = [torch.stack(instance['input_audios']) for instance in instances]
            input_audio_lens = [instance['input_audio_lens'] for instance in instances]
            input_audio_start = [instance["input_audio_start"] for instance in instances]
            
        else:
            input_audios = None
            input_audio_lens = None
            input_audio_start = None
            input_ids = None

        if instances[0]['decoder_audios'] is not None:
            decoder_audios = [torch.stack(instance['decoder_audios']) for instance in instances]
            decoder_audio_lens = [instance['decoder_audio_lens'] for instance in instances]
            decoder_audio_start = [instance["decoder_audio_start"] for instance in instances]
            y_mel = torch.cat(y_mel,dim=0)

            audio_decoder_st = torch.tensor(audio_decoder_st)
            audio_decoder_ed = torch.tensor(audio_decoder_ed)
        else:
            decoder_audios = None
            decoder_audio_lens = None
            decoder_audio_start = None
            y_mel = None
            audio_decoder_st = None
            audio_decoder_ed = None

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100)
        
        # import ipdb; ipdb.set_trace()
        
        #这里需要对频谱做padding
        
        
        
        # print(input_ids.size())
        # print(labels.size())

        # input_audios: Optional[torch.FloatTensor] = None,
        # input_audio_lens: Optional[torch.FloatTensor] = None,
        # input_audio_start = None,

        # decoder_audios: Optional[torch.FloatTensor] = None,
        # decoder_audio_lens: Optional[torch.FloatTensor] = None,
        # decoder_audio_start = None,
        
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            input_audios=input_audios,
            input_audio_lens=input_audio_lens,
            input_audio_start=input_audio_start,

            decoder_audios=decoder_audios,
            decoder_audio_lens=decoder_audio_lens,
            decoder_audio_start=decoder_audio_start,

            y_mel=y_mel,
            audio_decoder_st=audio_decoder_st,
            audio_decoder_ed=audio_decoder_ed

        )
        
        # print(batch)
        return batch