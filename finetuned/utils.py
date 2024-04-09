from typing import  Sequence, Any

from math import floor
from transformers import BertModel, BertTokenizer


#special tokens
PAD = "$"
MASK = "."
UNK = "?"
SEP = "|"
CLS = "*"

def is_whitespaced(seq: str
                   ) -> bool:
    """
    This function detects whether there is whitespace between characters in an input string
    """
    tok = list(seq)
    spaces = [t for t in tok if t.isspace()]
    if len(spaces) == floor(len(seq) / 2):
        return True
    return False

def get_pretrained_bert_tokenizer(path: str
                                  ) -> BertTokenizer:
    """Get the pretrained BERT tokenizer. This is a character level tokenizer of amino acids within the beta chain"""
    tok = BertTokenizer.from_pretrained(
        path,
        do_basic_tokenize=False,
        do_lower_case=False,
        tokenize_chinese_chars=False,
        unk_token=UNK,
        sep_token=SEP,
        pad_token=PAD,
        cls_token=CLS,
        mask_token=MASK,
        padding_side="right",
    )
    return tok



def insert_whitespace(seq: str) -> str:
    """ 
    Inserts whitespace between each amino acid in a beta chain sequence. 
    """
    return " ".join(list(seq))