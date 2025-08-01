# -*- coding: utf-8 -*-
# Time      :2025/3/29 11:17
# Author    :Hui Huang
import asyncio
import platform
import random
from typing import Literal, Optional, Callable, AsyncIterator
import soundfile as sf
import torch
import numpy as np
from ..llm import initialize_llm
from .utils import split_text, parse_multi_speaker_text, limit_concurrency
from functools import partial
from abc import ABC, abstractmethod
from ..logger import get_logger

logger = get_logger()


class Engine(ABC):

    @abstractmethod
    def list_roles(self) -> list[str]:
        ...

    @abstractmethod
    async def add_speaker(self, name: str, audio, reference_text: Optional[str] = None):
        ...

    @abstractmethod
    async def delete_speaker(self, name: str):
        ...

    @abstractmethod
    async def speak_async(
            self,
            name: str,
            text: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> np.ndarray:
        ...

    @abstractmethod
    async def speak_stream_async(
            self,
            name: str,
            text: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> AsyncIterator[np.ndarray]:
        yield  # type: ignore

    @abstractmethod
    async def clone_voice_async(
            self,
            text: str,
            reference_audio,
            reference_text: Optional[str] = None,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> np.ndarray:
        ...

    @abstractmethod
    async def clone_voice_stream_async(
            self,
            text: str,
            reference_audio,
            reference_text: Optional[str] = None,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> AsyncIterator[np.ndarray]:
        yield  # type: ignore

    @abstractmethod
    async def generate_voice_async(
            self,
            text: str,
            gender: Optional[Literal["female", "male"]] = "female",
            pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> np.ndarray:
        ...

    @abstractmethod
    async def generate_voice_stream_async(
            self,
            text: str,
            gender: Optional[Literal["female", "male"]] = "female",
            pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> AsyncIterator[np.ndarray]:
        yield  # type: ignore

    @abstractmethod
    async def multi_speak_async(
            self,
            text: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs
    ) -> np.ndarray:
        ...

    @abstractmethod
    async def multi_speak_stream_async(
            self,
            text: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs
    ) -> AsyncIterator[np.ndarray]:
        yield  # type: ignore


class BaseEngine(Engine):
    SAMPLE_RATE = 16000

    def __init__(
            self,
            llm_model_path: str,
            max_length: int = 32768,
            llm_device: Literal["cpu", "cuda", "mps", "auto"] | str = "auto",
            backend: Literal["vllm", "llama-cpp", "sglang", "torch", "mlx-lm"] = "torch",
            llm_attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            torch_dtype: Literal['float16', "bfloat16", 'float32', 'auto'] = "auto",
            llm_gpu_memory_utilization: Optional[float] = 0.6,
            cache_implementation: Optional[str] = None,
            llm_batch_size: int = 256,
            seed: int = 0,
            stop_tokens: Optional[list[str]] = None,
            stop_token_ids: Optional[list[int]] = None,
            **kwargs
    ):
        self.generator = initialize_llm(
            model_path=llm_model_path,
            backend=backend,
            max_length=max_length,
            device=self._auto_detect_device(llm_device),
            attn_implementation=llm_attn_implementation,
            torch_dtype=torch_dtype,
            gpu_memory_utilization=llm_gpu_memory_utilization,
            cache_implementation=cache_implementation,
            batch_size=llm_batch_size,
            seed=seed,
            stop_tokens=stop_tokens,
            stop_token_ids=stop_token_ids,
            **kwargs
        )
        self._batch_size = llm_batch_size
        
        # Store device information for debugging
        self.llm_device_info = self._auto_detect_device(llm_device)

    def get_device_summary(self) -> dict[str, str]:
        """Get a summary of device usage across all components."""
        summary = {
            "llm_device": self.llm_device_info,
            "tokenizer_device": getattr(self.audio_tokenizer, 'device', 'unknown'),
            "detokenizer_device": getattr(self.audio_detokenizer, 'device', 'unknown'),
        }
        
        # Get LLM backend device info if available
        if hasattr(self.generator, 'get_device_info'):
            summary["llm_backend_device"] = self.generator.get_device_info()
        elif hasattr(self.generator, 'device_info'):
            summary["llm_backend_device"] = self.generator.device_info
        else:
            summary["llm_backend_device"] = "unknown"
            
        return summary

    def list_roles(self) -> list[str]:
        raise NotImplementedError(f"List_roles not implemented for {self.__class__.__name__}")

    @classmethod
    def set_seed(cls, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @classmethod
    def _auto_detect_device(cls, device: str):
        """Auto-detect device with improved error handling and logging."""
        if not device:
            logger.warning("Device parameter is empty, defaulting to auto-detection")
            device = "auto"
            
        # Handle explicit device specifications
        if device in ["cpu", "cuda", "mps"] or device.startswith("cuda:"):
            logger.info(f"Using explicitly specified device: {device}")
            return device
            
        # Handle auto-detection
        if device == "auto":
            try:
                if torch.cuda.is_available():
                    detected_device = "cuda"
                    logger.info(f"Auto-detected CUDA device: {detected_device}")
                    return detected_device
                elif platform.system() == "Darwin" and torch.backends.mps.is_available():
                    detected_device = "mps"
                    logger.info(f"Auto-detected MPS device: {detected_device}")
                    return detected_device
                else:
                    detected_device = "cpu"
                    logger.info(f"Auto-detected CPU device: {detected_device}")
                    return detected_device
            except Exception as e:
                logger.error(f"Error during device auto-detection: {e}, falling back to CPU")
                return "cpu"
        
        # Handle unknown device strings - log warning and fallback
        logger.warning(f"Unknown device specification '{device}', falling back to CPU")
        return "cpu"

    def write_audio(self, audio: np.ndarray, filepath: str):
        sf.write(filepath, audio, self.SAMPLE_RATE, "PCM_16")

    def split_text(
            self,
            text: str,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None
    ) -> list[str]:
        tokenize_fn = partial(
            self.generator.tokenizer.encode,
            add_special_tokens=False,
            truncation=False,
            padding=False
        )
        return split_text(
            text, window_size,
            tokenize_fn=tokenize_fn,
            split_fn=split_fn,
            length_threshold=length_threshold
        )

    def _parse_multi_speak_text(self, text: str) -> list[dict[str, str]]:
        if len(self.list_roles()) == 0:
            msg = f"{self.__class__.__name__} 中角色库为空，无法实现多角色语音合成。"
            logger.error(msg)
            raise RuntimeError(msg)

        segments = parse_multi_speaker_text(text, self.list_roles())
        if len(segments) == 0:
            msg = f"多角色文本解析结果为空，请检查输入文本格式：{text}"
            logger.error(msg)
            raise RuntimeError(msg)

        return segments

    async def add_speaker(self, name: str, audio, reference_text: Optional[str] = None):
        raise NotImplementedError(f"add_speaker not implemented for {self.__class__.__name__}")

    async def delete_speaker(self, name: str):
        raise NotImplementedError(f"delete_speaker not implemented for {self.__class__.__name__}")

    async def speak_async(
            self,
            name: str,
            text: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> np.ndarray:
        raise NotImplementedError(f"Speak_async not implemented for {self.__class__.__name__}")

    async def speak_stream_async(
            self,
            name: str,
            text: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> AsyncIterator[np.ndarray]:
        yield NotImplementedError(f"speak_stream_async not implemented for {self.__class__.__name__}")

    async def clone_voice_async(
            self,
            text: str,
            reference_audio,
            reference_text: Optional[str] = None,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> np.ndarray:
        raise NotImplementedError(f"Clone_voice_async not implemented for {self.__class__.__name__}")

    async def clone_voice_stream_async(
            self,
            text: str,
            reference_audio,
            reference_text: Optional[str] = None,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> AsyncIterator[np.ndarray]:
        yield NotImplementedError(f"clone_voice_stream_async not implemented for {self.__class__.__name__}")

    async def generate_voice_async(
            self,
            text: str,
            gender: Optional[Literal["female", "male"]] = "female",
            pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> np.ndarray:
        raise NotImplementedError(f"generate_voice_async not implemented for {self.__class__.__name__}")

    async def generate_voice_stream_async(
            self,
            text: str,
            gender: Optional[Literal["female", "male"]] = "female",
            pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> AsyncIterator[np.ndarray]:
        yield NotImplementedError(f"generate_voice_stream_async not implemented for {self.__class__.__name__}")

    async def multi_speak_async(
            self,
            text: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs
    ) -> np.ndarray:
        """
        调用多角色共同合成语音。
        text (str): 待解析的文本，文本中各段台词前以 <角色名> 标识。
        如：<角色1>你好，欢迎来到我们的节目。<角色2>谢谢，我很高兴在这里。<角色3>大家好！
        """
        segments = self._parse_multi_speak_text(text)
        semaphore = asyncio.Semaphore(self._batch_size)  # 限制并发数，避免超长文本卡死
        limit_speak = limit_concurrency(semaphore)(self.speak_async)
        tasks = [
            asyncio.create_task(
                limit_speak(
                    name=segment['name'],
                    text=segment['text'],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    length_threshold=length_threshold,
                    window_size=window_size,
                    split_fn=split_fn,
                    **kwargs
                )
            ) for segment in segments]
        # 并发执行所有任务
        audios = await asyncio.gather(*tasks)
        audio = np.concatenate(audios, axis=0)
        return audio

    async def multi_speak_stream_async(
            self,
            text: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs
    ) -> AsyncIterator[np.ndarray]:
        segments = self._parse_multi_speak_text(text)

        for segment in segments:
            async for chunk in self.speak_stream_async(
                    name=segment['name'],
                    text=segment['text'],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    length_threshold=length_threshold,
                    window_size=window_size,
                    split_fn=split_fn,
                    **kwargs
            ):
                yield chunk
