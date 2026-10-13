"""A tiny *random-weight* laya.Agent for plumbing tests.

It is the real `laya.Agent` class (real tokenisation, option budget, forward, softmax and result
schema) wrapped around a 1-layer BERT with random weights and a throwaway word-level tokenizer.
It proves that this app talks to Laya's real API correctly. It says NOTHING about the quality of
the real Laya checkpoint.
"""
import string

import torch
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast

from laya.agent import Agent
from laya.common import DecisionModel


def make_tiny_agent(seed=0):
    torch.manual_seed(seed)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for tok in list(string.ascii_uppercase) + ["_", ":", ",", ".", "(", ")", "0", "1", "2", "3", "4", "5", "6", "7",
                                                "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
                                                "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
                                                "30", "of", "hit", "miss", "none", "Word", "pattern", "letters",
                                                "Incorrect", "guesses", "left", "Guesses", "so", "far", "Hangman",
                                                "question", "choice"]:
        vocab.setdefault(tok, len(vocab))
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
                                  sep_token="[SEP]", mask_token="[MASK]")
    cfg = AutoConfig.for_model("bert", hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=64, vocab_size=len(vocab), max_position_embeddings=1024)
    agent = object.__new__(Agent)
    agent.cfg = {"max_len": 512, "head_max_len": 192, "encoder": "tiny-random"}
    agent.tok = tok
    agent.model = DecisionModel(AutoModel.from_config(cfg), head_layers=1, n_act=2).eval()
    agent.device = torch.device("cpu")
    agent.dtype = torch.float32
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    return agent
