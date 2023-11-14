import os
import sys
sys.path.append('.')
import chinese
from symbols import *


_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def cleaned_text_to_sequence(cleaned_text, tones, language):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
  tone_start = language_tone_start_map[language]
  tones = [i + tone_start for i in tones]
  lang_id = language_id_map[language]
  lang_ids = [lang_id for i in phones]
  return phones, tones, lang_ids

# def get_bert(norm_text, word2ph, language):
#   from .chinese_bert import get_bert_feature as zh_bert
#   from .english_bert_mock import get_bert_feature as en_bert
#   lang_bert_func_map = {
#     'ZH': zh_bert,
#     'EN': en_bert
#   }
#   bert = lang_bert_func_map[language](norm_text, word2ph)
#   return bert


language_module_map = {
    'ZH': chinese
}


def clean_text(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph

def clean_text_bert(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    # bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, word2ph


def clean_text_bert(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert

def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)

if __name__ == '__main__':
    for file in os.listdir('/data/hypertext/sharpwang/dataset/leidian'):
        print(file)
    
    phones, tones, word2ph = clean_text_bert('我喜欢你','ZH')
    print(phones)
    print(tones)