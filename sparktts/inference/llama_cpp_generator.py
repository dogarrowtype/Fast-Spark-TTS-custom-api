# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/3/13 10:51
# Author  : Hui Huang
import os
from typing import Optional

from sparktts.inference.generator import Generator
from sparktts.inference.generic import GGUF_MODEL_NAME


class LlamaCPPGenerator(Generator):
    def __init__(
            self,
            model_path: str,
            gguf_model_file: Optional[str] = None,
            max_length: int = 32768,
            **kwargs
    ):
        from llama_cpp import Llama
        self.model = Llama(
            os.path.join(model_path, GGUF_MODEL_NAME) if gguf_model_file is None else gguf_model_file,
            n_ctx=max_length,
            **kwargs
        )
        # 不使用llama cpp 的 tokenizer
        super(LlamaCPPGenerator, self).__init__(
            tokenizer=model_path,
            max_length=max_length,
        )

    def generate(
            self,
            prompt: str,
            max_tokens: int = 1024,
            temperature: float = 0.6,
            top_p: float = 0.9,
            top_k: int = 50,
            **kwargs
    ) -> str:
        prompt_tokens = self.tokenize(prompt, self.max_length - max_tokens)
        completion_tokens = []
        for token in self.model.generate(
                prompt_tokens,
                top_k=top_k,
                top_p=top_p,
                temp=temperature,
                **kwargs
        ):
            if token in self.stop_token_ids:
                break
            if len(completion_tokens) + len(prompt_tokens) >= self.max_length:
                break
            completion_tokens.append(token)

        # Decode the generated tokens into text
        output = self.tokenizer.decode(completion_tokens)
        return output
