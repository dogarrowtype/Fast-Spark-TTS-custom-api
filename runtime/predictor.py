# -*- coding: utf-8 -*-
# Time      :2025/3/13 20:37
# Author    :Hui Huang
import os
import re
from typing import Optional, Literal, Tuple

import numpy as np
import torch
import soundfile as sf
from sparktts.utils.token_parser import TASK_TOKEN_MAP
from .audio_tokenizer import AudioTokenizer
from .vocoder import VoCoder


def process_prompt(
        text: str,
        prompt_text: Optional[str] = None,
        global_token_ids: torch.Tensor = None,
        semantic_token_ids: torch.Tensor = None,
) -> Tuple[str, torch.Tensor]:
    """
    Process input for voice cloning.

    Args:
        text: The text input to be converted to speech.
        prompt_text: Transcript of the prompt audio.
        global_token_ids: Global token IDs extracted from reference audio.
        semantic_token_ids: Semantic token IDs extracted from reference audio.

    Returns:
        Tuple containing the formatted input prompt and global token IDs.
    """
    # Convert global tokens to string format
    global_tokens = "".join(
        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
    )

    # Prepare the input tokens for the model
    if prompt_text is not None:
        # Include semantic tokens when prompt text is provided
        semantic_tokens = "".join(
            [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
        )

        inputs = [
            TASK_TOKEN_MAP["tts"],
            "<|start_content|>",
            prompt_text,
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
            "<|start_semantic_token|>",
            semantic_tokens,
        ]
    else:
        # Without prompt text, exclude semantic tokens
        inputs = [
            TASK_TOKEN_MAP["tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
        ]

    # Join all input components into a single string
    inputs = "".join(inputs)
    return inputs, global_token_ids


class AsyncFastSparkTTS:
    def __init__(
            self,
            model_path: str,
            max_length: int = 32768,
            gguf_model_file: Optional[str] = None,
            llm_device: Literal["cpu", "cuda"] | str = "cpu",
            audio_device: Literal["cpu", "cuda"] | str = "cpu",
            vocoder_device: Literal["cpu", "cuda"] | str = "cpu",
            engine: Literal["vllm", "llama-cpp", "sglang"] = "llama-cpp",
            wav2vec_attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            llm_gpu_memory_utilization: Optional[float] = 0.6,
            batch_size: int = 32,
            wait_timeout: float = 0.01,
            **kwargs,
    ):
        self.llm_device = llm_device
        self.audio_device = audio_device
        self.vocoder_device = vocoder_device

        if engine == "vllm":
            from .generator.vllm_generator import VllmGenerator
            self.generator = VllmGenerator(
                model_path=os.path.join(model_path, "LLM"),
                max_length=max_length,
                device=llm_device,
                max_num_seqs=batch_size,
                gpu_memory_utilization=llm_gpu_memory_utilization,
                **kwargs)
        elif engine == "llama-cpp":
            from .generator.llama_cpp_generator import LlamaCPPGenerator
            self.generator = LlamaCPPGenerator(
                model_path=os.path.join(model_path, "LLM"),
                gguf_model_file=gguf_model_file,
                max_length=max_length,
                **kwargs)
        elif engine == 'sglang':
            from .generator.sglang_generator import SglangGenerator
            self.generator = SglangGenerator(
                model_path=os.path.join(model_path, "LLM"),
                max_length=max_length,
                device=llm_device,
                gpu_memory_utilization=llm_gpu_memory_utilization,
                max_running_requests=batch_size,
                **kwargs)
        else:
            raise ValueError(f"Unknown backend: {engine}")

        self.audio_tokenizer = AudioTokenizer(
            model_path,
            device=audio_device,
            attn_implementation=wav2vec_attn_implementation,
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )
        self.vocoder = VoCoder(
            model_path=os.path.join(model_path, "BiCodec"),
            device=vocoder_device,
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )

    async def predict(
            self,
            reference_wav: np.ndarray,
            reference_wav_len: np.ndarray,
            reference_text: str,
            target_text: str,
            max_tokens: int = 1024,
            temperature: float = 0.6,
            top_p: float = 0.9,
            top_k: int = 50,
            **kwargs
    ) -> np.ndarray:
        audio_output = await self.audio_tokenizer.async_process(
            request={
                "reference_wav": reference_wav,
                "reference_wav_len": reference_wav_len
            })
        global_tokens, semantic_tokens = audio_output['global_tokens'], audio_output['semantic_tokens']
        prompt, global_token_ids = process_prompt(
            text=target_text,
            prompt_text=reference_text,
            global_token_ids=global_tokens,
            semantic_token_ids=semantic_tokens,
        )
        generated_output = await self.generator.async_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", generated_output)])
            .unsqueeze(0).to(torch.int32)
        )
        audio = await self.vocoder.async_process(
            request={
                "global_tokens": global_token_ids,
                "semantic_tokens": pred_semantic_ids,
            }
        )
        audio = audio['waveform'][0].cpu().numpy().astype(np.float32)
        return audio

    @classmethod
    async def prepare_inputs(
            cls,
            reference_wav_path: str,
            reference_text: str,
            target_text: str,
            max_tokens: int = 1024,
            temperature: float = 0.6,
            top_p: float = 0.9,
            top_k: int = 50,
            **kwargs
    ):
        waveform, sr = sf.read(reference_wav_path)
        assert sr == 16000, "sample rate hardcoded in server"

        lengths = np.array([len(waveform)], dtype=np.int32)
        samples = np.array(waveform, dtype=np.float32)
        samples = samples.reshape(1, -1).astype(np.float32)

        return dict(
            reference_wav=samples,
            reference_wav_len=lengths,
            reference_text=reference_text,
            target_text=target_text,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            **kwargs
        )

    async def inference(
            self,
            reference_wav_path: str,
            reference_text: str,
            target_text: str,
            max_tokens: int = 1024,
            temperature: float = 0.6,
            top_p: float = 0.9,
            top_k: int = 50,
            **kwargs) -> np.ndarray:

        inputs = await self.prepare_inputs(
            reference_wav_path=reference_wav_path,
            reference_text=reference_text,
            target_text=target_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )
        audio = await self.predict(
            **inputs
        )
        return audio
