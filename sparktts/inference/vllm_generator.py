# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/3/13 11:10
# Author  : Hui Huang
from sparktts.inference.generator import Generator


class VllmGenerator(Generator):
    def __init__(self, model_path: str, max_length: int = 32768, **kwargs):
        from vllm import LLM

        self.model = LLM(model=model_path, max_model_len=max_length, **kwargs)

        super(VllmGenerator, self).__init__(
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
        from vllm import SamplingParams
        prompt_tokens = self.tokenize(prompt, self.max_length - max_tokens)
        inputs = {"prompt_token_ids": prompt_tokens}
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_token_ids=self.stop_token_ids,
            max_tokens=max_tokens,
            **kwargs)
        output = self.model.generate(inputs, sampling_params)
        return output[0].outputs[0].text
