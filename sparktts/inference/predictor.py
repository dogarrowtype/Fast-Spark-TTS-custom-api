# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/3/13 10:50
# Author  : Hui Huang
import os
import re
import torch
from typing import Tuple, Literal, Optional
from pathlib import Path

from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP


class FastSparkTTS:
    def __init__(
            self,
            model_path: str,
            max_length: int = 32768,
            gguf_model_file: Optional[str] = None,
            llm_device: Literal["cpu", "cuda"] | str = "cpu",
            bicodec_device: Literal["cpu", "cuda"] | str = "cpu",
            backend: Literal["vllm", "llama-cpp"] = "llama-cpp",
            attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            **kwargs,
    ):
        self.llm_device = llm_device
        self.bicodec_device = torch.device(bicodec_device)

        if backend == "vllm":
            from .vllm_generator import VllmGenerator
            self.generator = VllmGenerator(
                model_path=os.path.join(model_path, "LLM"),
                max_length=max_length, **kwargs)
        elif backend == "llama-cpp":
            from .llama_cpp_generator import LlamaCPPGenerator
            self.generator = LlamaCPPGenerator(
                model_path=os.path.join(model_path, "LLM"),
                gguf_model_file=gguf_model_file,
                max_length=max_length, **kwargs)

        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.configs = load_config(os.path.join(model_path, "config.yaml"))
        self.sample_rate = self.configs["sample_rate"]
        self.audio_tokenizer = BiCodecTokenizer(
            model_path,
            device=self.bicodec_device,
            attn_implementation=attn_implementation)

    def process_prompt(
            self,
            text: str,
            prompt_speech_path: str,
            prompt_text: str = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.

        Return:
            Tuple[str, torch.Tensor]: Input prompt; global tokens
        """

        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
            prompt_speech_path
        )
        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
        )

        # Prepare the input tokens for the model
        if prompt_text is not None:
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
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)

        return inputs, global_token_ids

    @classmethod
    def process_prompt_control(
            cls,
            text: str,
            gender: Optional[Literal["female", "male"]] = None,
            pitch: Optional[Literal["very_low", " low", " moderate", " high", " very_high"]] = None,
            speed: Optional[Literal["very_low", " low", " moderate", " high", " very_high"]] = None,
    ):
        """
        Process input for voice creation.

        Args:
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            text (str): The text input to be converted to speech.

        Return:
            str: Input prompt
        """
        assert gender in GENDER_MAP.keys()
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join(
            [gender_tokens, pitch_label_tokens, speed_label_tokens]
        )

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)

    def inference(
            self,
            text: str,
            prompt_speech_path: Optional[str] = None,
            prompt_text: Optional[str] = None,
            gender: Optional[Literal["female", "male"]] = None,
            pitch: Optional[Literal["very_low", " low", " moderate", " high", " very_high"]] = None,
            speed: Optional[Literal["very_low", " low", " moderate", " high", " very_high"]] = None,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 1024,
            **kwargs
    ):
        if gender is not None:
            prompt = self.process_prompt_control(text, gender, pitch, speed)

        else:
            prompt, global_token_ids = self.process_prompt(
                text, prompt_speech_path, prompt_text
            )
        output = self.generator.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )
        # Extract semantic token IDs from the generated text
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", output)])
            .long()
            .unsqueeze(0)
        )

        if gender is not None:
            global_token_ids = (
                torch.tensor([int(token) for token in re.findall(r"bicodec_global_(\d+)", output)])
                .long()
                .unsqueeze(0)
                .unsqueeze(0)
            )

        with torch.no_grad():
            # Convert semantic tokens back to waveform
            wav = self.audio_tokenizer.detokenize(
                global_token_ids.to(self.bicodec_device).squeeze(0),
                pred_semantic_ids.to(self.bicodec_device),
            )

        return wav
