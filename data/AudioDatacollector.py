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
        
        audios = [torch.stack(instance['audios']) for instance in instances]
        audio_lens = [instance['audio_lens'] for instance in instances]
        audio_sts = [instance["audio_start"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100)
        # print(input_ids.size())
        # print(labels.size())
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            audios=audios,
            audio_lens=audio_lens,
            audio_start=audio_sts,
        )
        # print(batch)
        return batch