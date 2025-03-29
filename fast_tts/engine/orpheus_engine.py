# -*- coding: utf-8 -*-
# Time      :2025/3/29 15:36
# Author    :Hui Huang
import asyncio
import os.path
import re
from typing import Literal, Optional, Callable, AsyncIterator

import numpy as np
import torch

from .base_engine import BaseEngine
from ..audio import SnacDeTokenizer
from ..logger import get_logger
from .utils import contains_chinese, limit_concurrency

logger = get_logger()


class AsyncOrpheusEngine(BaseEngine):
    SAMPLE_RATE = 24000

    def __init__(
            self,
            model_path: str,
            max_length: int = 8192,
            snac_path: Optional[str] = None,
            llm_device: Literal["cpu", "cuda", "mps", "auto"] | str = "auto",
            detokenizer_device: Literal["cpu", "cuda", "mps", "auto"] | str = "auto",
            backend: Literal["vllm", "llama-cpp", "sglang", "torch"] = "torch",
            llm_attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            torch_dtype: Literal['float16', "bfloat16", 'float32', 'auto'] = "auto",
            llm_gpu_memory_utilization: Optional[float] = 0.8,  # snac模型显存暂用很小
            cache_implementation: Optional[str] = None,
            batch_size: int = 32,
            wait_timeout: float = 0.01,
            seed: int = 0,
            **kwargs
    ):
        self.seed = seed
        self.set_seed(seed)
        self.detokenizer = SnacDeTokenizer(
            snac_path if snac_path is not None else os.path.join(model_path, "snac"),
            device=detokenizer_device,
            batch_size=batch_size,
            wait_timeout=wait_timeout)

        self.speakers = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
        self._batch_size = batch_size

        super().__init__(
            llm_model_path=model_path,
            max_length=max_length,
            llm_device=llm_device,
            backend=backend,
            llm_attn_implementation=llm_attn_implementation,
            torch_dtype=torch_dtype,
            llm_gpu_memory_utilization=llm_gpu_memory_utilization,
            cache_implementation=cache_implementation,
            batch_size=batch_size,
            seed=seed,
            stop_token_ids=[49158],
            **kwargs
        )

    def list_roles(self) -> list[str]:
        roles = list(self.speakers)
        roles.sort()
        return roles

    def apply_prompt(
            self,
            text: str,
            name: Optional[str] = None
    ):
        if name is None:
            name = "tara"
        if name not in self.speakers:
            err_msg = f"{name} 角色不存在。"
            logger.error(err_msg)
            raise ValueError(err_msg)
        if contains_chinese(text):
            logger.warning("请注意：OrpheusTTS 目前还不支持中文。")
        prompt = f"<|audio|>{name}: {text}<|eot_id|><custom_token_4>"
        return prompt

    async def _convert_to_audio(self, multiframe: list[int]) -> np.ndarray | None:
        if len(multiframe) < 28:
            return None

        num_frames = len(multiframe) // 7
        # 截取完整帧的数据
        frame = multiframe[: num_frames * 7]
        # 将列表转换为 torch 张量，并重塑为 (num_frames, 7) 的形状
        frame_tensor = torch.tensor(frame, dtype=torch.int32).view(num_frames, 7)

        # 分别提取各个通道的 tokens
        # codes_0: 每帧的第 0 个元素，形状为 (num_frames,)
        codes_0 = frame_tensor[:, 0]
        # codes_1: 每帧的第 1 和第 4 个元素，形状为 (num_frames, 2) 后展平为 (num_frames*2,)
        codes_1 = frame_tensor[:, [1, 4]].reshape(-1)
        # codes_2: 每帧的第 2、3、5、6 个元素，形状为 (num_frames, 4) 后展平为 (num_frames*4,)
        codes_2 = frame_tensor[:, [2, 3, 5, 6]].reshape(-1)

        # 添加 batch 维度，使得形状分别变为 (1, num_frames)，(1, num_frames*2) 和 (1, num_frames*4)
        codes_0 = codes_0.unsqueeze(0)
        codes_1 = codes_1.unsqueeze(0)
        codes_2 = codes_2.unsqueeze(0)

        # 检查所有 token 是否均在 [0, 4096] 范围内
        if ((codes_0 < 0).any() or (codes_0 > 4096).any() or
                (codes_1 < 0).any() or (codes_1 > 4096).any() or
                (codes_2 < 0).any() or (codes_2 > 4096).any()):
            return None

        audio_hat = await self.detokenizer.detokenize_async([codes_0, codes_1, codes_2])
        # Process output
        audio = audio_hat["audio"][:, :, 2048:4096].detach().cpu().numpy()
        audio = (audio * 32767).astype(np.int16).reshape(1, -1)
        return audio.squeeze(0)

    async def _speak_stream(
            self,
            prompt: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            **kwargs
    ) -> AsyncIterator[np.ndarray]:
        buffer = []
        index = 0
        pattern = re.compile("<custom_token_(\d+)>")
        async for text_token in self.generator.async_stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                **kwargs
        ):
            audio_ids = pattern.findall(text_token)
            for audio_id in audio_ids:
                audio_id = int(audio_id)
                audio_id = int(audio_id) - 10 - ((index % 7) * 4096)

                if audio_id > 0:
                    buffer.append(audio_id)
                    index += 1

                    # Convert to audio when we have enough tokens
                    if index % 7 == 0 and index > 27:
                        buffer_to_proc = buffer[-28:]
                        audio_samples = await self._convert_to_audio(buffer_to_proc)
                        if audio_samples is not None:
                            yield audio_samples

    async def _speak(
            self,
            prompt: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            **kwargs) -> np.ndarray:
        buffer = []
        async for chunk in self._speak_stream(
                prompt=prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens,
                **kwargs
        ):
            buffer.append(chunk)
        return np.concatenate(buffer, axis=0)

    async def speak_stream_async(
            self,
            name: str,
            text: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> AsyncIterator[np.ndarray]:

        self.set_seed(seed=self.seed)
        segments = self.split_text(
            text=text,
            length_threshold=length_threshold,
            window_size=window_size,
            split_fn=split_fn
        )
        prompts = [self.apply_prompt(name=name, text=seg) for seg in segments]
        for prompt in prompts:
            async for audio in self._speak_stream(
                    prompt=prompt,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    **kwargs
            ):
                yield audio

    async def speak_async(
            self,
            name: str,
            text: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> np.ndarray:
        self.set_seed(seed=self.seed)
        segments = self.split_text(
            text=text,
            length_threshold=length_threshold,
            window_size=window_size,
            split_fn=split_fn
        )
        prompts = [self.apply_prompt(name=name, text=seg) for seg in segments]

        semaphore = asyncio.Semaphore(self._batch_size)  # 限制并发数，避免超长文本卡死
        limit_speak = limit_concurrency(semaphore)(self._speak)
        tasks = [
            asyncio.create_task(
                limit_speak(
                    prompt=prompt,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    **kwargs
                )
            ) for prompt in prompts
        ]
        # 并发执行所有任务
        audios = await asyncio.gather(*tasks)
        final_audio = np.concatenate(audios, axis=0)
        return final_audio

    async def clone_voice_stream_async(
            self,
            text: str,
            reference_audio,
            reference_text: Optional[str] = None,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> AsyncIterator[np.ndarray]:
        raise NotImplementedError("Clone voice is not implemented for Orpheus engine.")

    async def clone_voice_async(
            self,
            text: str,
            reference_audio,
            reference_text: Optional[str] = None,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> np.ndarray:
        raise NotImplementedError("Clone voice is not implemented for Orpheus engine.")

    async def generate_voice_stream_async(
            self,
            text: str,
            gender: Optional[Literal["female", "male"]] = "female",
            pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> AsyncIterator[np.ndarray]:
        raise NotImplementedError("Generate voice is not implemented for Orpheus engine.")

    async def generate_voice_async(
            self,
            text: str,
            gender: Optional[Literal["female", "male"]] = "female",
            pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> np.ndarray:
        raise NotImplementedError("Generate voice is not implemented for Orpheus engine.")
