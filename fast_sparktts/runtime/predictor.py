# -*- coding: utf-8 -*-
# Time      :2025/3/13 20:37
# Author    :Hui Huang
import asyncio
import os
import random
import re
from typing import Optional, Literal
import numpy as np
import torch
from .logger import get_logger
from .engine import Tokenizer, DeTokenizer
from .prompt import process_prompt, process_prompt_control, split_text

logger = get_logger()


def limit_concurrency(semaphore: asyncio.Semaphore):
    def decorator(func):
        async def wrapped(*args, **kwargs):
            async with semaphore:  # 在这里限制并发请求数
                return await func(*args, **kwargs)

        return wrapped

    return decorator


class AsyncFastSparkTTS:
    def __init__(
            self,
            model_path: str,
            max_length: int = 32768,
            llm_device: Literal["cpu", "cuda", "auto"] | str = "auto",
            tokenizer_device: Literal["cpu", "cuda", "auto"] | str = "auto",
            detokenizer_device: Literal["cpu", "cuda", "auto"] | str = "auto",
            engine: Literal["vllm", "llama-cpp", "sglang", "torch"] = "torch",
            wav2vec_attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            llm_attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            torch_dtype: Literal['float16', "bfloat16", 'float32', 'auto'] = "auto",
            llm_gpu_memory_utilization: Optional[float] = 0.6,
            cache_implementation: Optional[str] = None,
            batch_size: int = 32,
            wait_timeout: float = 0.01,
            seed: int = 0,
            **kwargs,
    ):
        """

        Args:
            model_path: 权重路径
            max_length: llm上下文最大长度
            gguf_model_file: llama cpp加载gguf模型文件，不传入则默认路径为 "{model_path}/LLM/model.gguf"
            llm_device: llm使用的device
            tokenizer_device: audio tokenizer使用的device
            detokenizer_device: audio detokenizer使用device
            engine: llm 后端类型
            wav2vec_attn_implementation: audio tokenizer中，wav2vec模型使用attn算子
            llm_gpu_memory_utilization: vllm和sglang暂用显存比例，单卡可降低该参数
            batch_size: 音频处理组件单批次处理的最大请求数。
            wait_timeout:
            **kwargs:
        """
        self.seed = seed
        self.set_seed(seed)

        self.audio_tokenizer = Tokenizer(
            model_path,
            device=self._auto_detect_device(tokenizer_device),
            attn_implementation=wav2vec_attn_implementation,
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )
        self.audio_detokenizer = DeTokenizer(
            model_path=model_path,
            device=self._auto_detect_device(detokenizer_device),
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )
        self.generator = self._init_llm(
            model_path=model_path,
            engine=engine,
            max_length=max_length,
            llm_device=llm_device,
            llm_attn_implementation=llm_attn_implementation,
            torch_dtype=torch_dtype,
            llm_gpu_memory_utilization=llm_gpu_memory_utilization,
            cache_implementation=cache_implementation,
            batch_size=batch_size,
            **kwargs
        )
        self.speakers = {}
        self._batch_size = batch_size

    @classmethod
    def set_seed(cls, seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _init_llm(
            self,
            model_path: str,
            engine: Literal["vllm", "llama-cpp", "sglang", "torch"] = "torch",
            max_length: int = 32768,
            llm_device: Literal["cpu", "cuda", "auto"] | str = "auto",
            llm_attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            torch_dtype: Literal['float16', "bfloat16", 'float32', 'auto'] = "auto",
            llm_gpu_memory_utilization: Optional[float] = 0.6,
            cache_implementation: Optional[str] = None,
            batch_size: int = 32,
            seed: int = 0,
            **kwargs
    ):
        llm_device = self._auto_detect_device(llm_device, is_sglang=bool(engine == "sglang"))
        llm_path = os.path.join(model_path, "LLM")

        if engine == "vllm":
            from .generator.vllm_generator import VllmGenerator
            generator = VllmGenerator(
                model_path=llm_path,
                max_length=max_length,
                device=llm_device,
                max_num_seqs=batch_size,
                gpu_memory_utilization=llm_gpu_memory_utilization,
                dtype=torch_dtype,
                seed=seed,
                **kwargs)
        elif engine == "llama-cpp":
            from .generator.llama_cpp_generator import LlamaCPPGenerator
            generator = LlamaCPPGenerator(
                model_path=llm_path,
                max_length=max_length,
                **kwargs)
        elif engine == 'sglang':
            from .generator.sglang_generator import SglangGenerator
            generator = SglangGenerator(
                model_path=llm_path,
                max_length=max_length,
                device=llm_device,
                gpu_memory_utilization=llm_gpu_memory_utilization,
                max_running_requests=batch_size,
                dtype=torch_dtype,
                random_seed=seed,
                **kwargs)
        elif engine == 'torch':
            from .generator.torch_generator import TorchGenerator

            generator = TorchGenerator(
                model_path=os.path.join(model_path, "LLM"),
                max_length=max_length,
                device=llm_device,
                attn_implementation=llm_attn_implementation,
                torch_dtype=torch_dtype,
                cache_implementation=cache_implementation,
                **kwargs)
        else:
            raise ValueError(f"Unknown backend: {engine}")
        return generator

    @classmethod
    def _auto_detect_device(cls, device: str, is_sglang: bool = False):
        if re.match("cuda:\d+", device):
            if is_sglang:
                logger.warning(
                    "sglang目前不支持指定GPU ID，将默认使用第一个GPU。您可以通过设置环境变量CUDA_VISIBLE_DEVICES=0 来指定GPU。")
                return "cuda"
            return device
        if device in ["cpu", "cuda"] or device.startswith("cuda"):
            return device
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def list_roles(self) -> list[str]:
        roles = list(self.speakers.keys())
        roles.sort()
        return roles

    async def _tokenize(
            self,
            audio
    ):
        output = await self.audio_tokenizer.tokenize_async(audio)
        return output

    async def _generate_audio_tokens(
            self,
            prompt: str,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            **kwargs
    ) -> dict[str, torch.Tensor | str]:
        generated_output = await self.generator.async_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )
        pred_semantic_tokens = [int(token) for token in re.findall(r"bicodec_semantic_(\d+)", generated_output)]
        if len(pred_semantic_tokens) == 0:
            err_msg = f"Semantic tokens 预测为空，prompt：{prompt}，llm output：{generated_output}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        pred_semantic_ids = (
            torch.tensor(pred_semantic_tokens).to(torch.int32)
        )

        output = {
            "semantic_tokens": pred_semantic_ids,
            "completion": generated_output,
        }

        global_tokens = [int(token) for token in re.findall(r"bicodec_global_(\d+)", generated_output)]
        if len(global_tokens) > 0:
            global_token_ids = (
                torch.tensor(global_tokens).unsqueeze(0).long()
            )
            output["global_tokens"] = global_token_ids

        return output

    async def _clone_voice_by_tokens(
            self,
            text: str,
            global_tokens: torch.Tensor,
            semantic_tokens: torch.Tensor,
            reference_text: Optional[str] = None,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            split: bool = False,
            window_size: int = 100,
            **kwargs) -> np.ndarray:
        if split:
            segments = split_text(text, window_size=window_size)
        else:
            segments = [text]

        async def clone_segment(segment):
            prompt, global_token_ids = process_prompt(
                text=segment,
                prompt_text=reference_text,
                global_token_ids=global_tokens,
                semantic_token_ids=semantic_tokens,
            )
            generated = await self._generate_audio_tokens(
                prompt=prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens,
                **kwargs
            )
            return generated

        if len(segments) > 1:
            semaphore = asyncio.Semaphore(self._batch_size)  # 限制并发数，避免超长文本卡死
            limit_clone_segment = limit_concurrency(semaphore)(clone_segment)
            tasks = [asyncio.create_task(limit_clone_segment(segment)) for segment in segments]
            # 并发执行所有任务
            generated_segments = await asyncio.gather(*tasks)
            semantic_token_ids = torch.cat(
                [generated["semantic_tokens"] for generated in generated_segments],
                dim=0
            )
        elif len(segments) == 1:
            semantic_token_ids = await clone_segment(segments[0])
            semantic_token_ids = semantic_token_ids['semantic_tokens']
        else:
            logger.error(f"请传入有效文本：{segments}")
            raise ValueError(f"请传入有效文本：{segments}")

        detokenizer_req = {
            "global_tokens": global_tokens,
            "semantic_tokens": semantic_token_ids
        }
        audio = await self.audio_detokenizer.detokenize_async(
            request=detokenizer_req
        )
        audio = audio["audio"][0].cpu().numpy().astype(np.float32)
        return audio

    async def add_speaker(self, name: str, audio, reference_text: Optional[str] = None):
        if name in self.speakers:
            logger.warning(f"{name} 音频已存在，将使用新的音频覆盖。")
        tokens = await self._tokenize(
            audio
        )
        self.speakers[name] = {
            "global_tokens": tokens['global_tokens'].detach().cpu(),
            "semantic_tokens": tokens['semantic_tokens'].detach().cpu(),
            "reference_text": reference_text
        }

    async def delete_speaker(self, name: str):
        if name not in self.speakers:
            logger.warning(f"{name} 角色不存在。")
            return
        self.speakers.pop(name)

    async def speak_async(
            self,
            name: str,
            text: str,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            split: bool = False,
            window_size: int = 100,
            **kwargs) -> np.ndarray:
        if name not in self.speakers:
            err_msg = f"{name} 角色不存在。"
            logger.error(err_msg)
            raise ValueError(err_msg)
        speaker = self.speakers[name]
        audio = await self._clone_voice_by_tokens(
            text=text,
            global_tokens=speaker['global_tokens'],
            semantic_tokens=speaker['semantic_tokens'],
            reference_text=speaker['reference_text'],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            split=split,
            window_size=window_size,
            **kwargs
        )
        return audio

    async def clone_voice_async(
            self,
            text: str,
            reference_audio,
            reference_text: Optional[str] = None,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            split: bool = False,
            window_size: int = 100,
            **kwargs) -> np.ndarray:

        tokens = await self._tokenize(
            reference_audio
        )
        audio = await self._clone_voice_by_tokens(
            text=text,
            global_tokens=tokens['global_tokens'],
            semantic_tokens=tokens['semantic_tokens'],
            reference_text=reference_text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            split=split,
            window_size=window_size,
            **kwargs
        )
        return audio

    async def generate_voice_async(
            self,
            text: str,
            gender: Optional[Literal["female", "male"]] = "female",
            pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            split: bool = False,  # 是否需要对文本切分，通常用于长文本场景
            window_size: int = 100,
            **kwargs) -> np.ndarray:

        if split:
            segments = split_text(text, window_size)
        else:
            segments = [text]

        async def generate_tokens(segment: str, acoustic_prompt: str):
            prompt = process_prompt_control(segment, gender, pitch, speed)
            if acoustic_prompt is not None:
                prompt += acoustic_prompt
            generated = await self._generate_audio_tokens(
                prompt=prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens,
                **kwargs
            )
            return generated

        # 使用第一段生成音色token，将其与后面片段一起拼接，使用相同音色token引导输出semantic tokens。
        first_prompt = process_prompt_control(segments[0], gender, pitch, speed)
        first_generated = await self._generate_audio_tokens(
            prompt=first_prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )
        if len(segments) > 1:
            # 提取关于音色的token部分
            acoustic_prompt = re.findall(
                r"(<\|start_acoustic_token\|>.*?<\|end_global_token\|>)",
                first_generated['completion'])
            if len(acoustic_prompt) > 0:
                acoustic_prompt = acoustic_prompt[0]
            else:
                acoustic_prompt = None

            semaphore = asyncio.Semaphore(self._batch_size)  # 限制并发数，避免超长文本卡死
            limit_generate_tokens = limit_concurrency(semaphore)(generate_tokens)

            tasks = [asyncio.create_task(limit_generate_tokens(segment, acoustic_prompt)) for segment in segments[1:]]
            # 并发执行所有任务
            generated_segments = await asyncio.gather(*tasks)
            generated_segments = [first_generated] + generated_segments
            semantic_tokens = torch.cat(
                [generated["semantic_tokens"] for generated in generated_segments],
                dim=0
            )
        else:
            semantic_tokens = first_generated['semantic_tokens']

        # 使用同一个global tokens尝试固定住音色。
        # 从这里可以看出，如果想要保持同一个音色，可以将acoustic_prompt和global_tokens保存下来即可
        detokenizer_req = {
            "global_tokens": first_generated['global_tokens'],
            "semantic_tokens": semantic_tokens
        }
        audio = await self.audio_detokenizer.detokenize_async(
            request=detokenizer_req
        )
        audio = audio["audio"][0].cpu().numpy().astype(np.float32)
        return audio

    def speak(
            self,
            name: str,
            text: str,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            split: bool = False,  # 是否需要对文本切分，通常用于长文本场景
            window_size: int = 100,
            **kwargs):
        return asyncio.run(
            self.speak_async(
                name=name,
                text=text,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens,
                split=split,
                window_size=window_size,
                **kwargs
            ))

    def clone_voice(
            self,
            text: str,
            reference_audio,
            reference_text: Optional[str] = None,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            split: bool = False,  # 是否需要对文本切分，通常用于长文本场景
            window_size: int = 100,
            **kwargs):
        return asyncio.run(
            self.clone_voice_async(
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens,
                split=split,
                window_size=window_size,
                **kwargs
            )
        )

    def generate_voice(
            self,
            text: str,
            gender: Optional[Literal["female", "male"]] = "female",
            pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 4096,
            split: bool = False,  # 是否需要对文本切分，通常用于长文本场景
            window_size: int = 100,
            **kwargs) -> np.ndarray:
        return asyncio.run(
            self.generate_voice_async(
                text=text,
                gender=gender,
                pitch=pitch,
                speed=speed,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens,
                split=split,
                window_size=window_size,
                **kwargs
            )
        )
