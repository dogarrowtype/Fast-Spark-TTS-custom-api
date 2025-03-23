# -*- coding: utf-8 -*-
# Time      :2025/3/23 20:00
# Author    :Hui Huang
import os
from typing import Literal
import torch
from .tokenizer import BaseModel
from .batch_processor import AsyncBatchEngine
from ...modules.encoder_decoder.feat_decoder import Decoder
from ...modules.encoder_decoder.wave_generator import WaveGenerator
from ...modules.speaker.speaker_encoder import SpeakerEncoder
from ...modules.vq.factorized_vector_quantize import FactorizedVectorQuantize


class DeTokenizerModel(BaseModel):
    def __init__(self, config):
        super().__init__()

        self.quantizer = FactorizedVectorQuantize(**config["quantizer"])
        self.prenet = Decoder(**config["prenet"])
        self.decoder = WaveGenerator(**config["decoder"])
        self.speaker_encoder = SpeakerEncoder(**config["speaker_encoder"])

    def forward(
            self,
            semantic_tokens: torch.Tensor,
            global_tokens: torch.Tensor
    ) -> torch.Tensor:
        z_q = self.quantizer.detokenize(semantic_tokens)
        d_vector = self.speaker_encoder.detokenize(global_tokens)
        x = self.prenet(z_q, d_vector)
        x = x + d_vector.unsqueeze(-1)
        wav_recon = self.decoder(x)
        return wav_recon


class DeTokenizer:
    def __init__(
            self,
            model_path: str,
            device: Literal["cpu", "cuda"] | str = "cpu",
            batch_size: int = 32,
            wait_timeout: float = 0.01):
        self.device = torch.device(device)
        self.model = DeTokenizerModel.from_pretrained(
            os.path.join(model_path, "BiCodec")
        ).to(self.device)

        self._batch_processor = AsyncBatchEngine(
            processing_function=self.batch_detokenize_async,
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )

    @torch.no_grad()
    def detokenize(
            self,
            semantic_tokens: torch.Tensor,
            global_tokens: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            semantic_tokens.to(self.device),
            global_tokens.to(self.device)
        )

    async def batch_detokenize_async(self, requests: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        semantic_tokens, global_tokens = [], []
        for request in requests:
            semantic_tokens.append(request["semantic_tokens"])
            global_tokens.append(request["global_tokens"])
        # Concatenate tokens for batch processing
        global_tokens = torch.stack(global_tokens, dim=0)
        semantic_tokens = torch.stack(semantic_tokens, dim=0)

        audios = self.detokenize(
            semantic_tokens=semantic_tokens,
            global_tokens=global_tokens
        )
        # Prepare responses
        responses = []
        for i in range(len(requests)):
            responses.append({
                "audio": audios[i],
            })

        return responses

    async def detokenize_async(self, request: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = await self._batch_processor.add_request(
            single_input=request
        )
        return output.get("feature")
