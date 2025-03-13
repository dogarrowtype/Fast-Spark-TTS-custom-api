# -*- coding: utf-8 -*-
# Time      :2025/3/13 19:57
# Author    :Hui Huang
import uuid
from typing import Literal, Optional

import numpy as np

from runtime.batch_processor import AsyncBatchEngine
from sparktts.models.audio_tokenizer import BiCodecTokenizer
import torch


class AudioTokenizer:
    def __init__(
            self,
            model_path: str,
            device: Literal["cpu", "cuda"] | str = "cpu",
            attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            batch_size: int = 32,
            wait_timeout: float = 0.01,
    ):
        self.device = torch.device(device)
        self.audio_tokenizer = BiCodecTokenizer(
            model_path,
            device=self.device,
            attn_implementation=attn_implementation)
        self._engine = AsyncBatchEngine(
            processing_function=self._batch_process,
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )

    @classmethod
    def get_ref_clip(cls, wav: np.ndarray) -> np.ndarray:
        """Extract reference audio clip for speaker embedding.

        Args:
            wav: Input waveform array

        Returns:
            Reference clip of fixed duration
        """
        SAMPLE_RATE = 16000
        REF_SEGMENT_DURATION = 6  # seconds
        LATENT_HOP_LENGTH = 320

        ref_segment_length = (
                int(SAMPLE_RATE * REF_SEGMENT_DURATION)
                // LATENT_HOP_LENGTH
                * LATENT_HOP_LENGTH
        )
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            # Repeat and truncate if input is too short
            repeat_times = ref_segment_length // wav_length + 1
            wav = np.tile(wav, repeat_times)

        return wav[:ref_segment_length]

    async def _batch_process(self, requests: list[dict[str, np.ndarray]]):
        reference_wav_list = []
        reference_wav_ref_clip_list = []

        # Process each request in batch
        for request in requests:
            # Extract input tensors
            wav_array = request["reference_wav"]
            wav_len = request["reference_wav_len"].item()

            # Prepare inputs
            wav = wav_array[:, :wav_len].squeeze(0)
            reference_wav_list.append(wav)

            wav_ref_clip = self.get_ref_clip(wav)
            reference_wav_ref_clip_list.append(torch.from_numpy(wav_ref_clip))

        # Batch process through tokenizer
        ref_wav_clip_tensor = torch.stack(reference_wav_ref_clip_list, dim=0)
        wav2vec2_features = self.audio_tokenizer.extract_wav2vec2_features(
            reference_wav_list)

        audio_tokenizer_input = {
            "ref_wav": ref_wav_clip_tensor.to(self.device),
            "feat": wav2vec2_features.to(self.device),
        }
        semantic_tokens, global_tokens = self.audio_tokenizer.model.tokenize(
            audio_tokenizer_input)

        # Prepare responses
        responses = []
        for i in range(len(requests)):
            responses.append({
                "global_tokens": global_tokens[i],
                "semantic_tokens": semantic_tokens[i]
            })

        return responses

    async def _async_process(
            self,
            request: dict[str, np.ndarray],
            request_id: str
    ) -> list[float]:
        embedding = await self._engine.add_request(
            single_input=request, request_id=request_id
        )
        return embedding.get("feature")

    async def async_process(
            self,
            request: dict[str, np.ndarray],
            request_id: Optional[str] = None):
        if request_id is None:
            request_id = str(uuid.uuid4().hex)
        output = await self._async_process(
            request=request,
            request_id=request_id)

        return output


