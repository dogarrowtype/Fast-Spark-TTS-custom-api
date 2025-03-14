# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/3/13 10:57
# Author  : Hui Huang
from typing import List
from transformers import AutoTokenizer
import uuid


class Generator:

    def __init__(self, tokenizer, max_length: int):
        self.max_length = max_length
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        stop_tokens = ["<|im_end|>"]
        stop_token_ids = self.tokenizer.convert_tokens_to_ids(stop_tokens)
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids = stop_token_ids + [self.tokenizer.eos_token_id]
        self.stop_token_ids = stop_token_ids

    def tokenize(self, text: str, max_length: int) -> List[int]:
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False
        )
        tokens = tokens[-max_length:]
        return tokens

    @classmethod
    async def random_uid(cls):
        return str(uuid.uuid4().hex)

    async def async_generate(
            self,
            prompt: str,
            max_tokens: int = 1024,
            temperature: float = 0.6,
            top_p: float = 0.9,
            top_k: int = 50,
            **kwargs
    ) -> str:
        raise NotImplementedError("generate method not implemented")
